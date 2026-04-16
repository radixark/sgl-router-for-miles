use serde_json::json;
use smg::protocols::generate::GenerateRequest;

#[test]
fn test_generate_extension_fields_roundtrip() {
    let json_with_extensions = json!({
        "text": "Hello world",
        "return_hidden_states": true,
        "return_routed_experts": true,
        "routed_experts_start_len": 10,
        "return_prompt_token_ids": true,
        "require_reasoning": true,
        "routed_dp_rank": 2,
        "disagg_prefill_dp_rank": 1,
        "data_parallel_rank": 3,
        "routing_key": "key-abc",
        "max_dynamic_patch": 12,
        "min_dynamic_patch": 1,
        "stream": false
    });

    let req: GenerateRequest =
        serde_json::from_value(json_with_extensions).expect("should deserialize");
    assert!(req.return_hidden_states);
    assert!(req.return_routed_experts);
    assert_eq!(req.routed_experts_start_len, 10);
    assert!(req.return_prompt_token_ids);
    assert!(req.require_reasoning);
    assert_eq!(req.routed_dp_rank, Some(2));
    assert_eq!(req.disagg_prefill_dp_rank, Some(1));
    assert_eq!(req.data_parallel_rank, Some(3));
    assert_eq!(req.routing_key.as_deref(), Some("key-abc"));
    assert_eq!(req.max_dynamic_patch, Some(12));
    assert_eq!(req.min_dynamic_patch, Some(1));

    let serialized = serde_json::to_value(&req).expect("should serialize");
    assert_eq!(serialized["return_hidden_states"], true);
    assert_eq!(serialized["return_routed_experts"], true);
    assert_eq!(serialized["routed_experts_start_len"], 10);
    assert_eq!(serialized["return_prompt_token_ids"], true);
    assert_eq!(serialized["require_reasoning"], true);
    assert_eq!(serialized["routed_dp_rank"], 2);
    assert_eq!(serialized["disagg_prefill_dp_rank"], 1);
    assert_eq!(serialized["data_parallel_rank"], 3);
    assert_eq!(serialized["routing_key"], "key-abc");
    assert_eq!(serialized["max_dynamic_patch"], 12);
    assert_eq!(serialized["min_dynamic_patch"], 1);
}

#[test]
fn test_generate_extension_fields_defaults() {
    let json_minimal = json!({
        "text": "Hello world",
        "stream": false
    });

    let req: GenerateRequest =
        serde_json::from_value(json_minimal).expect("should deserialize");
    assert!(!req.return_hidden_states);
    assert!(!req.return_routed_experts);
    assert_eq!(req.routed_experts_start_len, 0);
    assert!(!req.return_prompt_token_ids);
    assert!(!req.require_reasoning);
    assert!(req.routed_dp_rank.is_none());
    assert!(req.disagg_prefill_dp_rank.is_none());
    assert!(req.data_parallel_rank.is_none());
    assert!(req.routing_key.is_none());
    assert!(req.max_dynamic_patch.is_none());
    assert!(req.min_dynamic_patch.is_none());
}

#[test]
fn test_generate_optional_fields_omitted_when_none() {
    let json_minimal = json!({
        "text": "Hello world",
        "stream": false
    });

    let req: GenerateRequest =
        serde_json::from_value(json_minimal).expect("should deserialize");
    let serialized = serde_json::to_value(&req).expect("should serialize");

    let omitted = [
        "routed_dp_rank", "disagg_prefill_dp_rank", "data_parallel_rank",
        "routing_key", "max_dynamic_patch", "min_dynamic_patch",
    ];
    for field in omitted {
        assert!(
            serialized.get(field).is_none(),
            "{} should be omitted when None",
            field
        );
    }
}
