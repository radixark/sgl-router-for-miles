//! Data Parallel (DP) information discovery step.

use async_trait::async_trait;
use tracing::debug;
use crate::workflow::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::discover_metadata::get_server_info;
use crate::core::{steps::workflow_data::LocalWorkerWorkflowData, UNKNOWN_MODEL_ID};

/// DP (Data Parallel) information for a worker.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DpInfo {
    pub dp_size: usize,
    pub model_id: String,
}

/// Get DP info for a worker URL.
pub async fn get_dp_info(url: &str, api_key: Option<&str>) -> Result<DpInfo, String> {
    let info = get_server_info(url, api_key).await?;

    let dp_size = routing_dp_size(info.dp_size, info.attn_dp_size)
        .map_err(|e| format!("{} in response from {}", e, url))?;

    let model_id = info
        .model_id
        .filter(|s| !s.is_empty())
        .or(info.served_model_name.filter(|s| !s.is_empty()))
        .or_else(|| {
            info.model_path
                .and_then(|path| path.split('/').next_back().map(|s| s.to_string()))
        })
        .unwrap_or_else(|| UNKNOWN_MODEL_ID.to_string());

    Ok(DpInfo { dp_size, model_id })
}

/// Number of DP ranks requests can be routed to.
fn routing_dp_size(dp_size: Option<usize>, attn_dp_size: Option<usize>) -> Result<usize, String> {
    if dp_size.is_none() && attn_dp_size.is_none() {
        return Err("No dp_size".to_string());
    }
    // Newer SGLang separates model replicas from attention-DP groups; older
    // servers report their routing width in dp_size alone.
    dp_size
        .unwrap_or(1)
        .checked_mul(attn_dp_size.unwrap_or(1))
        .filter(|&width| width > 0)
        .ok_or_else(|| "DP sizes must be positive, with a product that fits usize".to_string())
}

/// Step 2b: Discover DP (Data Parallel) information (only for DP-aware workers).
pub struct DiscoverDPInfoStep;

#[async_trait]
impl StepExecutor<LocalWorkerWorkflowData> for DiscoverDPInfoStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<LocalWorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let config = &context.data.config;

        if !config.dp_aware {
            debug!(
                "Worker {} is not DP-aware, skipping DP discovery",
                config.url
            );
            return Ok(StepResult::Success);
        }

        debug!("Discovering DP info for {} (DP-aware)", config.url);

        let dp_info = get_dp_info(&config.url, config.api_key.as_deref())
            .await
            .map_err(|e| WorkflowError::StepFailed {
                step_id: StepId::new("discover_dp_info"),
                message: format!("Failed to get DP info: {}", e),
            })?;

        debug!(
            "Discovered DP size {} for {} (model: {})",
            dp_info.dp_size, config.url, dp_info.model_id
        );

        context.data.dp_info = Some(dp_info);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::routing_dp_size;

    #[test]
    fn test_attn_dp_groups_are_routable_ranks() {
        // `--dp-size 8 --enable-dp-attention` resolves to dp_size=1, attn_dp_size=8.
        assert_eq!(routing_dp_size(Some(1), Some(8)), Ok(8));
    }

    #[test]
    fn test_dp_size_alone_is_the_width() {
        assert_eq!(routing_dp_size(Some(4), None), Ok(4));
    }

    #[test]
    fn test_attn_dp_size_alone_is_the_width() {
        assert_eq!(routing_dp_size(None, Some(8)), Ok(8));
    }

    #[test]
    fn test_missing_sizes_are_refused() {
        assert!(routing_dp_size(None, None).is_err());
    }

    #[test]
    fn test_zero_width_is_refused() {
        assert!(routing_dp_size(Some(0), Some(8)).is_err());
    }
}
