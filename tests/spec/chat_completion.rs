use serde_json::json;
use smg::protocols::{
    chat::{ChatCompletionRequest, ChatMessage, MessageContent},
    common::{
        Function, FunctionCall, FunctionChoice, StreamOptions, Tool, ToolChoice, ToolChoiceValue,
        ToolReference,
    },
    validated::Normalizable,
};
use validator::Validate;

// Deprecated fields normalization tests

#[test]
fn test_max_tokens_normalizes_to_max_completion_tokens() {
    #[allow(deprecated)]
    let mut req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: MessageContent::Text("hello".to_string()),
            name: None,
        }],
        max_tokens: Some(100),
        max_completion_tokens: None,
        ..Default::default()
    };

    req.normalize();
    assert_eq!(
        req.max_completion_tokens,
        Some(100),
        "max_tokens should be copied to max_completion_tokens"
    );
    #[allow(deprecated)]
    {
        assert!(
            req.max_tokens.is_none(),
            "Deprecated field should be cleared"
        );
    }
    assert!(
        req.validate().is_ok(),
        "Should be valid after normalization"
    );
}

#[test]
fn test_max_completion_tokens_takes_precedence() {
    #[allow(deprecated)]
    let mut req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: MessageContent::Text("hello".to_string()),
            name: None,
        }],
        max_tokens: Some(100),
        max_completion_tokens: Some(200),
        ..Default::default()
    };

    req.normalize();
    assert_eq!(
        req.max_completion_tokens,
        Some(200),
        "max_completion_tokens should take precedence"
    );
    assert!(
        req.validate().is_ok(),
        "Should be valid after normalization"
    );
}

#[test]
fn test_functions_normalizes_to_tools() {
    #[allow(deprecated)]
    let mut req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: MessageContent::Text("hello".to_string()),
            name: None,
        }],
        functions: Some(vec![Function {
            name: "test_func".to_string(),
            description: Some("Test function".to_string()),
            parameters: json!({}),
            strict: None,
        }]),
        tools: None,
        ..Default::default()
    };

    req.normalize();
    assert!(req.tools.is_some(), "functions should be migrated to tools");
    assert_eq!(req.tools.as_ref().unwrap().len(), 1);
    assert_eq!(req.tools.as_ref().unwrap()[0].function.name, "test_func");
    #[allow(deprecated)]
    {
        assert!(
            req.functions.is_none(),
            "Deprecated field should be cleared"
        );
    }
    assert!(
        req.validate().is_ok(),
        "Should be valid after normalization"
    );
}

#[test]
fn test_function_call_normalizes_to_tool_choice() {
    #[allow(deprecated)]
    let mut req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: MessageContent::Text("hello".to_string()),
            name: None,
        }],
        function_call: Some(FunctionCall::None),
        tool_choice: None,
        ..Default::default()
    };

    req.normalize();
    assert!(
        req.tool_choice.is_some(),
        "function_call should be migrated to tool_choice"
    );
    assert!(matches!(
        req.tool_choice,
        Some(ToolChoice::Value(ToolChoiceValue::None))
    ));
    #[allow(deprecated)]
    {
        assert!(
            req.function_call.is_none(),
            "Deprecated field should be cleared"
        );
    }
    assert!(
        req.validate().is_ok(),
        "Should be valid after normalization"
    );
}

#[test]
fn test_function_call_function_variant_normalizes() {
    #[allow(deprecated)]
    let mut req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: MessageContent::Text("hello".to_string()),
            name: None,
        }],
        function_call: Some(FunctionCall::Function {
            name: "my_function".to_string(),
        }),
        tool_choice: None,
        tools: Some(vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "my_function".to_string(),
                description: None,
                parameters: json!({}),
                strict: None,
            },
        }]),
        ..Default::default()
    };

    req.normalize();
    assert!(
        req.tool_choice.is_some(),
        "function_call should be migrated to tool_choice"
    );
    match &req.tool_choice {
        Some(ToolChoice::Function { function, .. }) => {
            assert_eq!(function.name, "my_function");
        }
        _ => panic!("Expected ToolChoice::Function variant"),
    }
    #[allow(deprecated)]
    {
        assert!(
            req.function_call.is_none(),
            "Deprecated field should be cleared"
        );
    }
    assert!(
        req.validate().is_ok(),
        "Should be valid after normalization"
    );
}

// Stream options validation tests

#[test]
fn test_stream_options_requires_stream_enabled() {
    let req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: MessageContent::Text("hello".to_string()),
            name: None,
        }],
        stream: false,
        stream_options: Some(StreamOptions {
            include_usage: Some(true),
        }),
        ..Default::default()
    };

    let result = req.validate();
    assert!(
        result.is_err(),
        "Should reject stream_options when stream is false"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("stream_options") && err.contains("stream") && err.contains("enabled"),
        "Error should mention stream dependency: {}",
        err
    );
}

#[test]
fn test_stream_options_valid_when_stream_enabled() {
    let req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: MessageContent::Text("hello".to_string()),
            name: None,
        }],
        stream: true,
        stream_options: Some(StreamOptions {
            include_usage: Some(true),
        }),
        ..Default::default()
    };

    let result = req.validate();
    assert!(
        result.is_ok(),
        "Should accept stream_options when stream is true"
    );
}

#[test]
fn test_no_stream_options_valid_when_stream_disabled() {
    let req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: MessageContent::Text("hello".to_string()),
            name: None,
        }],
        stream: false,
        stream_options: None,
        ..Default::default()
    };

    let result = req.validate();
    assert!(
        result.is_ok(),
        "Should accept no stream_options when stream is false"
    );
}

// Tool choice validation tests
#[test]
fn test_tool_choice_function_not_found() {
    let req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: MessageContent::Text("hello".to_string()),
            name: None,
        }],
        tools: Some(vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: json!({}),
                strict: None,
            },
        }]),
        tool_choice: Some(ToolChoice::Function {
            function: FunctionChoice {
                name: "nonexistent_function".to_string(),
            },
            tool_type: "function".to_string(),
        }),
        ..Default::default()
    };

    let result = req.validate();
    assert!(result.is_err(), "Should reject nonexistent function name");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("function 'nonexistent_function' not found"),
        "Error should mention the missing function: {}",
        err
    );
}

#[test]
fn test_tool_choice_function_exists_valid() {
    let req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: MessageContent::Text("hello".to_string()),
            name: None,
        }],
        tools: Some(vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: json!({}),
                strict: None,
            },
        }]),
        tool_choice: Some(ToolChoice::Function {
            function: FunctionChoice {
                name: "get_weather".to_string(),
            },
            tool_type: "function".to_string(),
        }),
        ..Default::default()
    };

    let result = req.validate();
    assert!(result.is_ok(), "Should accept existing function name");
}

#[test]
fn test_tool_choice_allowed_tools_invalid_mode() {
    let req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: MessageContent::Text("hello".to_string()),
            name: None,
        }],
        tools: Some(vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: json!({}),
                strict: None,
            },
        }]),
        tool_choice: Some(ToolChoice::AllowedTools {
            mode: "invalid_mode".to_string(),
            tools: vec![ToolReference::Function {
                name: "get_weather".to_string(),
            }],
            tool_type: "function".to_string(),
        }),
        ..Default::default()
    };

    let result = req.validate();
    assert!(result.is_err(), "Should reject invalid mode");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("must be 'auto' or 'required'"),
        "Error should mention valid modes: {}",
        err
    );
}

#[test]
fn test_tool_choice_allowed_tools_valid_mode_auto() {
    let req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: MessageContent::Text("hello".to_string()),
            name: None,
        }],
        tools: Some(vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: json!({}),
                strict: None,
            },
        }]),
        tool_choice: Some(ToolChoice::AllowedTools {
            mode: "auto".to_string(),
            tools: vec![ToolReference::Function {
                name: "get_weather".to_string(),
            }],
            tool_type: "function".to_string(),
        }),
        ..Default::default()
    };

    let result = req.validate();
    assert!(result.is_ok(), "Should accept 'auto' mode");
}

#[test]
fn test_tool_choice_allowed_tools_valid_mode_required() {
    let req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: MessageContent::Text("hello".to_string()),
            name: None,
        }],
        tools: Some(vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: json!({}),
                strict: None,
            },
        }]),
        tool_choice: Some(ToolChoice::AllowedTools {
            mode: "required".to_string(),
            tools: vec![ToolReference::Function {
                name: "get_weather".to_string(),
            }],
            tool_type: "function".to_string(),
        }),
        ..Default::default()
    };

    let result = req.validate();
    assert!(result.is_ok(), "Should accept 'required' mode");
}

#[test]
fn test_tool_choice_allowed_tools_tool_not_found() {
    let req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: MessageContent::Text("hello".to_string()),
            name: None,
        }],
        tools: Some(vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: json!({}),
                strict: None,
            },
        }]),
        tool_choice: Some(ToolChoice::AllowedTools {
            mode: "auto".to_string(),
            tools: vec![ToolReference::Function {
                name: "nonexistent_tool".to_string(),
            }],
            tool_type: "function".to_string(),
        }),
        ..Default::default()
    };

    let result = req.validate();
    assert!(result.is_err(), "Should reject nonexistent tool name");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("tool 'nonexistent_tool' not found"),
        "Error should mention the missing tool: {}",
        err
    );
}

#[test]
fn test_tool_choice_allowed_tools_multiple_tools_valid() {
    let req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: MessageContent::Text("hello".to_string()),
            name: None,
        }],
        tools: Some(vec![
            Tool {
                tool_type: "function".to_string(),
                function: Function {
                    name: "get_weather".to_string(),
                    description: Some("Get weather".to_string()),
                    parameters: json!({}),
                    strict: None,
                },
            },
            Tool {
                tool_type: "function".to_string(),
                function: Function {
                    name: "get_time".to_string(),
                    description: Some("Get time".to_string()),
                    parameters: json!({}),
                    strict: None,
                },
            },
        ]),
        tool_choice: Some(ToolChoice::AllowedTools {
            mode: "auto".to_string(),
            tools: vec![
                ToolReference::Function {
                    name: "get_weather".to_string(),
                },
                ToolReference::Function {
                    name: "get_time".to_string(),
                },
            ],
            tool_type: "function".to_string(),
        }),
        ..Default::default()
    };

    let result = req.validate();
    assert!(result.is_ok(), "Should accept all valid tool references");
}

#[test]
fn test_tool_choice_allowed_tools_one_invalid_among_valid() {
    let req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: MessageContent::Text("hello".to_string()),
            name: None,
        }],
        tools: Some(vec![
            Tool {
                tool_type: "function".to_string(),
                function: Function {
                    name: "get_weather".to_string(),
                    description: Some("Get weather".to_string()),
                    parameters: json!({}),
                    strict: None,
                },
            },
            Tool {
                tool_type: "function".to_string(),
                function: Function {
                    name: "get_time".to_string(),
                    description: Some("Get time".to_string()),
                    parameters: json!({}),
                    strict: None,
                },
            },
        ]),
        tool_choice: Some(ToolChoice::AllowedTools {
            mode: "auto".to_string(),
            tools: vec![
                ToolReference::Function {
                    name: "get_weather".to_string(),
                },
                ToolReference::Function {
                    name: "nonexistent_tool".to_string(),
                },
            ],
            tool_type: "function".to_string(),
        }),
        ..Default::default()
    };

    let result = req.validate();
    assert!(
        result.is_err(),
        "Should reject if any tool reference is invalid"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("tool 'nonexistent_tool' not found"),
        "Error should mention the missing tool: {}",
        err
    );
}

// SGLang extension fields serde round-trip tests

#[test]
fn test_sglang_extension_fields_roundtrip() {
    let json_with_extensions = json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "return_hidden_states": true,
        "return_routed_experts": true,
        "return_cached_tokens_details": true,
        "return_prompt_token_ids": true,
        "return_meta_info": true,
        "input_ids": [1, 2, 3, 42, 100],
        "stop_regex": ["\\d+", "end"],
        "custom_logit_processor": "serialized_processor",
        "custom_params": {"key": "value"},
        "max_dynamic_patch": 12,
        "min_dynamic_patch": 1,
        "rid": "req-123",
        "extra_key": "salt-abc",
        "cache_salt": "salt-xyz",
        "priority": 5,
        "bootstrap_host": "10.0.0.1",
        "bootstrap_port": 8080,
        "bootstrap_room": 42,
        "routed_dp_rank": 2,
        "disagg_prefill_dp_rank": 1,
        "data_parallel_rank": 3
    });

    let req: ChatCompletionRequest =
        serde_json::from_value(json_with_extensions).expect("should deserialize");
    assert!(req.return_hidden_states);
    assert!(req.return_routed_experts);
    assert!(req.return_cached_tokens_details);
    assert!(req.return_prompt_token_ids);
    assert!(req.return_meta_info);
    assert_eq!(req.input_ids, Some(vec![1, 2, 3, 42, 100]));
    assert_eq!(req.max_dynamic_patch, Some(12));
    assert_eq!(req.min_dynamic_patch, Some(1));
    assert_eq!(req.priority, Some(5));
    assert_eq!(req.routed_dp_rank, Some(2));
    assert_eq!(req.disagg_prefill_dp_rank, Some(1));
    assert_eq!(req.data_parallel_rank, Some(3));

    let serialized = serde_json::to_value(&req).expect("should serialize");
    assert_eq!(serialized["return_hidden_states"], true);
    assert_eq!(serialized["return_routed_experts"], true);
    assert_eq!(serialized["return_cached_tokens_details"], true);
    assert_eq!(serialized["return_prompt_token_ids"], true);
    assert_eq!(serialized["return_meta_info"], true);
    assert_eq!(serialized["input_ids"], json!([1, 2, 3, 42, 100]));
    assert_eq!(serialized["stop_regex"], json!(["\\d+", "end"]));
    assert_eq!(serialized["custom_logit_processor"], "serialized_processor");
    assert_eq!(serialized["custom_params"], json!({"key": "value"}));
    assert_eq!(serialized["max_dynamic_patch"], 12);
    assert_eq!(serialized["min_dynamic_patch"], 1);
    assert_eq!(serialized["rid"], "req-123");
    assert_eq!(serialized["extra_key"], "salt-abc");
    assert_eq!(serialized["cache_salt"], "salt-xyz");
    assert_eq!(serialized["priority"], 5);
    assert_eq!(serialized["bootstrap_host"], "10.0.0.1");
    assert_eq!(serialized["bootstrap_port"], 8080);
    assert_eq!(serialized["bootstrap_room"], 42);
    assert_eq!(serialized["routed_dp_rank"], 2);
    assert_eq!(serialized["disagg_prefill_dp_rank"], 1);
    assert_eq!(serialized["data_parallel_rank"], 3);
}

#[test]
fn test_sglang_extension_fields_default_values() {
    let json_minimal = json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}]
    });

    let req: ChatCompletionRequest =
        serde_json::from_value(json_minimal).expect("should deserialize");
    assert!(!req.return_hidden_states);
    assert!(!req.return_routed_experts);
    assert!(!req.return_cached_tokens_details);
    assert!(!req.return_prompt_token_ids);
    assert!(!req.return_meta_info);
    assert!(req.input_ids.is_none());
    assert!(req.stop_regex.is_none());
    assert!(req.custom_logit_processor.is_none());
    assert!(req.custom_params.is_none());
    assert!(req.max_dynamic_patch.is_none());
    assert!(req.min_dynamic_patch.is_none());
    assert!(req.rid.is_none());
    assert!(req.extra_key.is_none());
    assert!(req.cache_salt.is_none());
    assert!(req.priority.is_none());
    assert!(req.bootstrap_host.is_none());
    assert!(req.bootstrap_port.is_none());
    assert!(req.bootstrap_room.is_none());
    assert!(req.routed_dp_rank.is_none());
    assert!(req.disagg_prefill_dp_rank.is_none());
    assert!(req.data_parallel_rank.is_none());
}

#[test]
fn test_sglang_optional_fields_omitted_when_none() {
    let req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: MessageContent::Text("hello".to_string()),
            name: None,
        }],
        ..Default::default()
    };

    let serialized = serde_json::to_value(&req).expect("should serialize");
    let omitted = [
        "input_ids", "stop_regex", "custom_logit_processor", "custom_params",
        "max_dynamic_patch", "min_dynamic_patch", "rid", "extra_key", "cache_salt",
        "priority", "bootstrap_host", "bootstrap_port", "bootstrap_room",
        "routed_dp_rank", "disagg_prefill_dp_rank", "data_parallel_rank",
    ];
    for field in omitted {
        assert!(
            serialized.get(field).is_none(),
            "{} should be omitted when None",
            field
        );
    }
}
