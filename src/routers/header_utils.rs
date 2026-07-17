use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, HeaderValue},
    response::Response,
};
use http::header::HeaderName;

use crate::routers::error;

static HEADER_TARGET_WORKER: HeaderName = HeaderName::from_static("x-smg-target-worker");
static HEADER_ROUTING_KEY: HeaderName = HeaderName::from_static("x-smg-routing-key");

fn extract_header_value<'a>(headers: Option<&'a HeaderMap>, name: &HeaderName) -> Option<&'a str> {
    headers
        .and_then(|h| h.get(name))
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
}

pub fn extract_target_worker(headers: Option<&HeaderMap>) -> Option<&str> {
    extract_header_value(headers, &HEADER_TARGET_WORKER)
}

pub fn extract_routing_key(headers: Option<&HeaderMap>) -> Option<&str> {
    extract_header_value(headers, &HEADER_ROUTING_KEY)
}

/// Under `consistent_hashing` and `manual` policies a keyless request silently loses
/// session affinity, so reject it unless the router is started with
/// `--allow-requests-without-routing-key`.
pub fn missing_routing_key_response(
    policy_names: &[&str],
    headers: Option<&HeaderMap>,
    allow_requests_without_routing_key: bool,
) -> Option<Response> {
    if allow_requests_without_routing_key {
        return None;
    }
    if !policy_names
        .iter()
        .any(|name| matches!(*name, "consistent_hashing" | "manual"))
    {
        return None;
    }
    if extract_routing_key(headers).is_some() || extract_target_worker(headers).is_some() {
        return None;
    }
    Some(error::bad_request(
        "missing_routing_key",
        "the routing policy requires an X-SMG-Routing-Key header on every request; \
         attach a routing key or start the router with --allow-requests-without-routing-key",
    ))
}

/// Copy request headers to a Vec of name-value string pairs
/// Used for forwarding headers to backend workers
pub fn copy_request_headers(req: &Request<Body>) -> Vec<(String, String)> {
    req.headers()
        .iter()
        .filter_map(|(name, value)| {
            // Convert header value to string, skipping non-UTF8 headers
            value
                .to_str()
                .ok()
                .map(|v| (name.to_string(), v.to_string()))
        })
        .collect()
}

/// Convert headers from reqwest Response to axum HeaderMap
/// Filters out hop-by-hop headers that shouldn't be forwarded
pub fn preserve_response_headers(reqwest_headers: &HeaderMap) -> HeaderMap {
    let mut headers = HeaderMap::new();

    for (name, value) in reqwest_headers.iter() {
        // Skip hop-by-hop headers that shouldn't be forwarded
        // Use eq_ignore_ascii_case to avoid string allocation
        if should_forward_header_no_alloc(name.as_str()) {
            // The original name and value are already valid, so we can just clone them
            headers.insert(name.clone(), value.clone());
        }
    }

    headers
}

/// Determine if a header should be forwarded without allocating (case-insensitive)
fn should_forward_header_no_alloc(name: &str) -> bool {
    // List of headers that should NOT be forwarded (hop-by-hop headers)
    // Use eq_ignore_ascii_case to avoid to_lowercase() allocation
    !(name.eq_ignore_ascii_case("connection")
        || name.eq_ignore_ascii_case("keep-alive")
        || name.eq_ignore_ascii_case("proxy-authenticate")
        || name.eq_ignore_ascii_case("proxy-authorization")
        || name.eq_ignore_ascii_case("te")
        || name.eq_ignore_ascii_case("trailers")
        || name.eq_ignore_ascii_case("transfer-encoding")
        || name.eq_ignore_ascii_case("upgrade")
        || name.eq_ignore_ascii_case("content-encoding")
        || name.eq_ignore_ascii_case("host"))
}

/// Apply headers to a reqwest request builder, filtering out headers that shouldn't be forwarded
/// or that will be set automatically by reqwest
pub fn apply_request_headers(
    headers: &HeaderMap,
    mut request_builder: reqwest::RequestBuilder,
    skip_content_headers: bool,
) -> reqwest::RequestBuilder {
    // Always forward Authorization header first if present
    if let Some(auth) = headers
        .get("authorization")
        .or_else(|| headers.get("Authorization"))
    {
        request_builder = request_builder.header("Authorization", auth.clone());
    }

    // Forward other headers, filtering out problematic ones
    // Use eq_ignore_ascii_case to avoid to_lowercase() allocation per header
    for (key, value) in headers.iter() {
        let key_str = key.as_str();

        // Skip headers that:
        // - Are set automatically by reqwest (content-type, content-length for POST/PUT)
        // - We already handled (authorization)
        // - Are hop-by-hop headers (connection, transfer-encoding)
        // - Should not be forwarded (host)
        let should_skip = key_str.eq_ignore_ascii_case("authorization") // Already handled above
            || key_str.eq_ignore_ascii_case("host")
            || key_str.eq_ignore_ascii_case("connection")
            || key_str.eq_ignore_ascii_case("transfer-encoding")
            || key_str.eq_ignore_ascii_case("keep-alive")
            || key_str.eq_ignore_ascii_case("te")
            || key_str.eq_ignore_ascii_case("trailers")
            || key_str.eq_ignore_ascii_case("accept-encoding")
            || key_str.eq_ignore_ascii_case("upgrade")
            || (skip_content_headers
                && (key_str.eq_ignore_ascii_case("content-type")
                    || key_str.eq_ignore_ascii_case("content-length")));

        if !should_skip {
            request_builder = request_builder.header(key.clone(), value.clone());
        }
    }

    request_builder
}

/// API provider types for provider-specific header handling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiProvider {
    Anthropic,
    Xai,
    OpenAi,
    Gemini,
    Generic,
}

impl ApiProvider {
    /// Detect provider type from URL
    pub fn from_url(url: &str) -> Self {
        if url.contains("anthropic") {
            ApiProvider::Anthropic
        } else if url.contains("x.ai") {
            ApiProvider::Xai
        } else if url.contains("openai.com") {
            ApiProvider::OpenAi
        } else if url.contains("googleapis.com") {
            ApiProvider::Gemini
        } else {
            ApiProvider::Generic
        }
    }
}

/// Apply provider-specific headers to request
pub fn apply_provider_headers(
    mut req: reqwest::RequestBuilder,
    url: &str,
    auth_header: Option<&HeaderValue>,
) -> reqwest::RequestBuilder {
    let provider = ApiProvider::from_url(url);

    match provider {
        ApiProvider::Anthropic => {
            // Anthropic requires x-api-key instead of Authorization
            // Extract Bearer token and use as x-api-key
            if let Some(auth) = auth_header {
                if let Ok(auth_str) = auth.to_str() {
                    let api_key = auth_str.strip_prefix("Bearer ").unwrap_or(auth_str);
                    req = req
                        .header("x-api-key", api_key)
                        .header("anthropic-version", "2023-06-01");
                }
            }
        }
        ApiProvider::Gemini | ApiProvider::Xai | ApiProvider::OpenAi | ApiProvider::Generic => {
            // Standard OpenAI-compatible: use Authorization header as-is
            if let Some(auth) = auth_header {
                req = req.header("Authorization", auth);
            }
        }
    }

    req
}

/// Extract auth header with passthrough semantics.
///
/// Passthrough mode: User's Authorization header takes priority.
/// Fallback: Worker's API key is used only if user didn't provide auth.
///
/// This enables use cases where:
/// 1. Users send their own API keys (multi-tenant, BYOK)
/// 2. Router has a default key for users who don't provide one
pub fn extract_auth_header(
    headers: Option<&HeaderMap>,
    worker_api_key: &Option<String>,
) -> Option<HeaderValue> {
    // Passthrough: Try user's auth header first
    let user_auth = headers.and_then(|h| {
        h.get("authorization")
            .or_else(|| h.get("Authorization"))
            .cloned()
    });

    // Return user's auth if provided, otherwise use worker's API key
    user_auth.or_else(|| {
        worker_api_key
            .as_ref()
            .and_then(|k| HeaderValue::from_str(&format!("Bearer {}", k)).ok())
    })
}

#[inline]
pub fn should_forward_request_header(name: &str) -> bool {
    const REQUEST_ID_PREFIX: &str = "x-request-id-";

    name.eq_ignore_ascii_case("authorization")
        || name.eq_ignore_ascii_case("x-request-id")
        || name.eq_ignore_ascii_case("x-correlation-id")
        || name.eq_ignore_ascii_case("traceparent")
        || name.eq_ignore_ascii_case("tracestate")
        || name.eq_ignore_ascii_case("x-smg-routing-key")
        || name
            .get(..REQUEST_ID_PREFIX.len())
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case(REQUEST_ID_PREFIX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_header_value_returns_value() {
        let mut headers = HeaderMap::new();
        headers.insert("x-smg-routing-key", "test-key".parse().unwrap());
        assert_eq!(extract_routing_key(Some(&headers)), Some("test-key"));
    }

    #[test]
    fn test_extract_header_value_returns_none_for_missing() {
        let headers = HeaderMap::new();
        assert_eq!(extract_routing_key(Some(&headers)), None);
    }

    #[test]
    fn test_extract_header_value_returns_none_for_empty() {
        let mut headers = HeaderMap::new();
        headers.insert("x-smg-routing-key", "".parse().unwrap());
        assert_eq!(extract_routing_key(Some(&headers)), None);
    }

    #[test]
    fn test_extract_header_value_returns_none_for_none_headers() {
        assert_eq!(extract_routing_key(None), None);
    }

    #[test]
    fn test_extract_target_worker() {
        let mut headers = HeaderMap::new();
        headers.insert("x-smg-target-worker", "2".parse().unwrap());
        assert_eq!(extract_target_worker(Some(&headers)), Some("2"));
    }

    #[test]
    fn test_extract_target_worker_missing() {
        let headers = HeaderMap::new();
        assert_eq!(extract_target_worker(Some(&headers)), None);
    }

    #[test]
    fn test_should_forward_request_header_whitelist() {
        assert!(should_forward_request_header("authorization"));
        assert!(should_forward_request_header("Authorization"));
        assert!(should_forward_request_header("AUTHORIZATION"));
        assert!(should_forward_request_header("x-request-id"));
        assert!(should_forward_request_header("X-Request-Id"));
        assert!(should_forward_request_header("x-correlation-id"));
        assert!(should_forward_request_header("X-Correlation-ID"));
        assert!(should_forward_request_header("traceparent"));
        assert!(should_forward_request_header("Traceparent"));
        assert!(should_forward_request_header("tracestate"));
        assert!(should_forward_request_header("Tracestate"));
        assert!(should_forward_request_header("x-request-id-user"));
        assert!(should_forward_request_header("X-Request-ID-Span"));
        assert!(should_forward_request_header("x-request-id-123"));
        assert!(should_forward_request_header("x-smg-routing-key"));
        assert!(should_forward_request_header("X-SMG-Routing-Key"));
    }

    #[test]
    fn test_should_forward_request_header_blocked() {
        assert!(!should_forward_request_header("content-type"));
        assert!(!should_forward_request_header("Content-Type"));
        assert!(!should_forward_request_header("content-length"));
        assert!(!should_forward_request_header("host"));
        assert!(!should_forward_request_header("Host"));
        assert!(!should_forward_request_header("connection"));
        assert!(!should_forward_request_header("transfer-encoding"));
        assert!(!should_forward_request_header("accept"));
        assert!(!should_forward_request_header("accept-encoding"));
        assert!(!should_forward_request_header("user-agent"));
        assert!(!should_forward_request_header("cookie"));
        assert!(!should_forward_request_header("x-custom-header"));
        assert!(!should_forward_request_header("x-api-key"));
    }

    #[test]
    fn test_missing_routing_key_rejected_for_key_policies() {
        let empty = HeaderMap::new();
        for policy in ["consistent_hashing", "manual"] {
            assert!(missing_routing_key_response(&[policy], Some(&empty), false).is_some());
        }
        // empty header value counts as missing
        let mut headers = HeaderMap::new();
        headers.insert("x-smg-routing-key", "".parse().unwrap());
        assert!(missing_routing_key_response(&["manual"], Some(&headers), false).is_some());
        // PD mode: one key-based policy is enough to require the key
        assert!(missing_routing_key_response(&["random", "manual"], Some(&empty), false).is_some());
    }

    #[test]
    fn test_missing_routing_key_allowed_cases() {
        let empty = HeaderMap::new();
        // escape hatch disables the check
        assert!(missing_routing_key_response(&["manual"], Some(&empty), true).is_none());
        // policies that do not route by key
        for policy in ["cache_aware", "round_robin", "random", "power_of_two"] {
            assert!(missing_routing_key_response(&[policy], Some(&empty), false).is_none());
        }
        // routing key present
        let mut headers = HeaderMap::new();
        headers.insert("x-smg-routing-key", "abc".parse().unwrap());
        assert!(missing_routing_key_response(&["manual"], Some(&headers), false).is_none());
        // explicit target worker is an alternative routing directive
        let mut headers = HeaderMap::new();
        headers.insert("x-smg-target-worker", "0".parse().unwrap());
        assert!(
            missing_routing_key_response(&["consistent_hashing"], Some(&headers), false).is_none()
        );
    }
}
