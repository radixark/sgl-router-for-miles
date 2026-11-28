//! Capacity-aware load balancing policy
//!
//! Routes each request to the worker with the fewest in-flight requests per unit of
//! capacity. Meant for pools of unequal workers, e.g. rollout engines on different GPU
//! generations or vendors, where plain least-requests gives every worker the same share.

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use async_trait::async_trait;
use tracing::debug;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::{config::CapacitySignal, core::Worker};

/// Capacity-aware policy
///
/// Picks the healthy worker minimizing `(in_flight + 1) / capacity`. Capacity comes from
/// the configured [`CapacitySignal`] and is refreshed by the load monitor. If any healthy
/// worker has no capacity yet, every worker falls back to plain least-requests, so a
/// partial capacity map never skews the pool.
#[derive(Debug)]
pub struct CapacityAwarePolicy {
    signal: CapacitySignal,
    /// Worker URL -> KV-cache capacity in tokens
    kv_capacities: RwLock<HashMap<String, f64>>,
}

impl CapacityAwarePolicy {
    pub fn new(signal: CapacitySignal) -> Self {
        Self {
            signal,
            kv_capacities: RwLock::new(HashMap::new()),
        }
    }

    pub fn signal(&self) -> CapacitySignal {
        self.signal
    }

    /// Capacity weight for each worker in `indices`, or None if any worker is missing one
    fn weights(&self, workers: &[Arc<dyn Worker>], indices: &[usize]) -> Option<Vec<f64>> {
        let kv = self.kv_capacities.read().ok()?;
        indices
            .iter()
            .map(|&i| match self.signal {
                CapacitySignal::Kv => kv.get(workers[i].url()).copied(),
            })
            .map(|w| w.filter(|w| *w > 0.0))
            .collect()
    }
}

#[async_trait]
impl LoadBalancingPolicy for CapacityAwarePolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);
        if healthy_indices.is_empty() {
            return None;
        }

        let selected = match self.weights(workers, &healthy_indices) {
            Some(weights) => healthy_indices
                .iter()
                .zip(weights)
                .map(|(&i, w)| (i, (workers[i].load() as f64 + 1.0) / w))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(i, _)| i)?,
            None => *healthy_indices.iter().min_by_key(|&&i| workers[i].load())?,
        };

        debug!(
            "Capacity-aware ({}) selection: {} with {} in flight",
            self.signal.as_str(),
            workers[selected].url(),
            workers[selected].load()
        );

        workers[selected].increment_processed();
        Some(selected)
    }

    fn name(&self) -> &'static str {
        "capacity_aware"
    }

    fn update_capacities(&self, capacities: &HashMap<String, f64>) {
        if let Ok(mut cached) = self.kv_capacities.write() {
            *cached = capacities.clone();
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for CapacityAwarePolicy {
    fn default() -> Self {
        Self::new(CapacitySignal::default())
    }
}

/// KV-cache capacity in tokens from a worker's `/get_server_info` response.
///
/// SGLang reports `max_total_num_tokens` per data-parallel replica, so the worker's
/// capacity is that times `dp_size`.
pub(crate) fn kv_capacity_from_server_info(info: &serde_json::Value) -> Option<f64> {
    let per_replica = info
        .get("max_total_num_tokens")
        .or_else(|| {
            info.get("internal_states")
                .and_then(|s| s.get(0))
                .and_then(|s| s.get("max_total_num_tokens"))
        })
        .and_then(|v| v.as_f64())?;
    let dp_size = info.get("dp_size").and_then(|v| v.as_f64()).unwrap_or(1.0);
    let capacity = per_replica * dp_size.max(1.0);
    (capacity > 0.0).then_some(capacity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    fn worker(url: &str, in_flight: usize) -> Arc<dyn Worker> {
        let w = BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .build();
        for _ in 0..in_flight {
            w.increment_load();
        }
        Arc::new(w)
    }

    fn caps(entries: &[(&str, f64)]) -> HashMap<String, f64> {
        entries.iter().map(|(u, c)| (u.to_string(), *c)).collect()
    }

    #[tokio::test]
    async fn test_capacity_aware_weights_by_kv() {
        let policy = CapacityAwarePolicy::new(CapacitySignal::Kv);
        // Big worker already has more in flight, but per token of KV it is still the emptier one.
        let workers = vec![
            worker("http://big:8000", 50),
            worker("http://small:8000", 10),
        ];
        policy.update_capacities(&caps(&[
            ("http://big:8000", 3_000_000.0),
            ("http://small:8000", 400_000.0),
        ]));

        let info = SelectWorkerInfo::default();
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
    }

    #[tokio::test]
    async fn test_capacity_aware_split_follows_capacity() {
        let policy = CapacityAwarePolicy::new(CapacitySignal::Kv);
        let workers = vec![worker("http://a:8000", 0), worker("http://b:8000", 0)];
        policy.update_capacities(&caps(&[("http://a:8000", 3.0), ("http://b:8000", 1.0)]));

        // Simulate requests that stay in flight: the split should approach 3:1.
        let info = SelectWorkerInfo::default();
        let mut counts = [0usize; 2];
        for _ in 0..400 {
            let idx = policy.select_worker(&workers, &info).await.unwrap();
            workers[idx].increment_load();
            counts[idx] += 1;
        }
        assert!((295..=305).contains(&counts[0]), "split was {:?}", counts);
    }

    #[tokio::test]
    async fn test_capacity_aware_falls_back_when_capacity_missing() {
        let policy = CapacityAwarePolicy::new(CapacitySignal::Kv);
        let workers = vec![worker("http://big:8000", 50), worker("http://new:8000", 10)];
        // Only one worker reported a capacity: fall back to least-requests for all.
        policy.update_capacities(&caps(&[("http://big:8000", 3_000_000.0)]));

        let info = SelectWorkerInfo::default();
        assert_eq!(policy.select_worker(&workers, &info).await, Some(1));
    }

    #[tokio::test]
    async fn test_capacity_aware_skips_unhealthy() {
        let policy = CapacityAwarePolicy::new(CapacitySignal::Kv);
        let workers = vec![worker("http://a:8000", 0), worker("http://b:8000", 0)];
        policy.update_capacities(&caps(&[("http://a:8000", 3.0), ("http://b:8000", 1.0)]));
        workers[0].set_healthy(false);

        let info = SelectWorkerInfo::default();
        for _ in 0..10 {
            assert_eq!(policy.select_worker(&workers, &info).await, Some(1));
        }

        workers[1].set_healthy(false);
        assert_eq!(policy.select_worker(&workers, &info).await, None);
    }

    #[test]
    fn test_kv_capacity_from_server_info() {
        let info = serde_json::json!({"max_total_num_tokens": 400704, "dp_size": 1});
        assert_eq!(kv_capacity_from_server_info(&info), Some(400704.0));

        let info = serde_json::json!({"max_total_num_tokens": 1000, "dp_size": 4});
        assert_eq!(kv_capacity_from_server_info(&info), Some(4000.0));

        let info = serde_json::json!({"internal_states": [{"max_total_num_tokens": 500}]});
        assert_eq!(kv_capacity_from_server_info(&info), Some(500.0));

        assert_eq!(kv_capacity_from_server_info(&serde_json::json!({})), None);
    }
}
