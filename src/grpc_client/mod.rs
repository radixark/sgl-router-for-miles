pub mod sglang_scheduler;
pub mod vllm_engine;

// Export both clients
// Re-export proto modules with explicit names
pub use sglang_scheduler::{proto as sglang_proto, SglangSchedulerClient};
pub use vllm_engine::{proto as vllm_proto, VllmEngineClient};

/// Trait for injecting trace context into gRPC metadata.
/// Implementors can propagate distributed tracing headers across gRPC calls.
pub trait TraceInjector: Send + Sync {
    fn inject(
        &self,
        metadata: &mut tonic::metadata::MetadataMap,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}
