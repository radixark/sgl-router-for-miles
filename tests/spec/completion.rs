use serde_json::json;
use smg::protocols::completion::CompletionRequest;

#[test]
fn test_completion_sglang_extension_fields_roundtrip() {
    let json_with_extensions = json!({
        "model": "test-model",
        "prompt": "hello",
        "return_hidden_states": true,
        "return_routed_experts": true,
        "routed_experts_start_len": 10
    });

    let req: CompletionRequest =
        serde_json::from_value(json_with_extensions).expect("should deserialize");
    assert!(req.return_hidden_states);
    assert!(req.return_routed_experts);
    assert_eq!(req.routed_experts_start_len, 10);

    let serialized = serde_json::to_value(&req).expect("should serialize");
    assert_eq!(serialized["return_hidden_states"], true);
    assert_eq!(serialized["return_routed_experts"], true);
    assert_eq!(serialized["routed_experts_start_len"], 10);
}

#[test]
fn test_completion_sglang_extension_fields_default_values() {
    let json_minimal = json!({
        "model": "test-model",
        "prompt": "hello"
    });

    let req: CompletionRequest =
        serde_json::from_value(json_minimal).expect("should deserialize");
    assert!(!req.return_hidden_states);
    assert!(!req.return_routed_experts);
    assert_eq!(req.routed_experts_start_len, 0);
}
