//! HTTP client for the OpenAI Responses API (Deep Research).
//!
//! Provides `DeepResearchClient` with methods to create and retrieve
//! deep research jobs. Uses reqwest with rustls for HTTPS.

use anyhow::{Context, Result};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use tracing::{debug, instrument};

use super::types::{
    CreateRequest, DeepResearchModel, InputMessage, ReasoningConfig, Response, SearchContextSize,
    Tool, UserLocation,
};

const OPENAI_API_BASE: &str = "https://api.openai.com/v1";

/// Client for interacting with the OpenAI Responses API.
#[derive(Debug, Clone)]
pub struct DeepResearchClient {
    http: reqwest::Client,
    base_url: String,
}

impl DeepResearchClient {
    /// Create a new client with the given API key.
    pub fn new(api_key: &str) -> Result<Self> {
        Self::with_base_url(api_key, OPENAI_API_BASE)
    }

    /// Create a new client with a custom base URL (for testing or proxies).
    pub fn with_base_url(api_key: &str, base_url: &str) -> Result<Self> {
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {api_key}"))
                .context("invalid API key characters")?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let http = reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .context("failed to build HTTP client")?;

        Ok(Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }

    /// Create a new deep research job.
    ///
    /// This sends a POST to `/v1/responses` with `background: true`.
    /// Returns immediately with the response object in queued/in_progress state.
    #[instrument(skip(self, query), fields(model = %model))]
    pub async fn create(
        &self,
        query: &str,
        model: DeepResearchModel,
        options: CreateOptions,
    ) -> Result<Response> {
        if let Some(size) = &options.search_context_size {
            anyhow::ensure!(
                matches!(size, SearchContextSize::Medium),
                "Deep research only supports search_context_size 'medium'; omit the option or use 'medium'"
            );
        }

        let mut tools: Vec<Tool> = Vec::new();

        // Web search is always included for deep research.
        let user_location =
            if options.country.is_some() || options.city.is_some() || options.region.is_some() {
                let mut loc = UserLocation::new();
                if let Some(c) = &options.country {
                    loc = loc.with_country(c.as_str());
                }
                if let Some(c) = &options.city {
                    loc = loc.with_city(c.as_str());
                }
                if let Some(r) = &options.region {
                    loc = loc.with_region(r.as_str());
                }
                Some(loc)
            } else {
                None
            };

        tools.push(Tool::WebSearchPreview {
            user_location,
            search_context_size: options.search_context_size,
        });

        // Optionally include code interpreter.
        if options.code_interpreter {
            tools.push(Tool::CodeInterpreter {});
        }

        let mut input = Vec::new();

        // Optional developer instructions.
        if let Some(instructions) = &options.instructions {
            input.push(InputMessage {
                role: "developer".to_string(),
                content: instructions.clone(),
            });
        }

        // The user query.
        input.push(InputMessage {
            role: "user".to_string(),
            content: query.to_string(),
        });

        let request = CreateRequest {
            model: model.to_string(),
            input,
            tools,
            reasoning: ReasoningConfig::default(),
            background: true,
            instructions: None, // Using input-level developer message instead.
            store: options.store,
            metadata: options.metadata,
        };

        debug!("creating deep research job");

        let resp = self
            .http
            .post(format!("{}/responses", self.base_url))
            .json(&request)
            .send()
            .await
            .context("failed to send create request")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenAI API error (HTTP {status}): {body}");
        }

        resp.json::<Response>()
            .await
            .context("failed to parse create response")
    }

    /// Retrieve the current state of a deep research job.
    ///
    /// This sends a GET to `/v1/responses/{id}`.
    /// Poll this until `response.is_terminal()` returns true.
    #[instrument(skip(self))]
    pub async fn retrieve(&self, response_id: &str) -> Result<Response> {
        debug!("retrieving response");

        let resp = self
            .http
            .get(format!("{}/responses/{}", self.base_url, response_id))
            .send()
            .await
            .context("failed to send retrieve request")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenAI API error (HTTP {status}): {body}");
        }

        resp.json::<Response>()
            .await
            .context("failed to parse retrieve response")
    }

    /// Poll a response until it reaches a terminal state.
    ///
    /// Uses exponential backoff starting at 5s, capping at 30s.
    /// Deep research jobs typically take 5-30 minutes.
    #[instrument(skip(self))]
    pub async fn poll_until_complete(&self, response_id: &str) -> Result<Response> {
        let mut interval_secs = 5u64;
        let max_interval = 30u64;

        loop {
            let response = self.retrieve(response_id).await?;

            if response.is_terminal() {
                return Ok(response);
            }

            debug!(
                status = %response.status,
                steps = response.research_step_count(),
                next_poll_secs = interval_secs,
                "research in progress"
            );

            tokio::time::sleep(std::time::Duration::from_secs(interval_secs)).await;
            interval_secs = (interval_secs * 2).min(max_interval);
        }
    }
}

/// Options for creating a deep research job.
#[derive(Debug, Clone, Default)]
pub struct CreateOptions {
    /// ISO 3166-1 alpha-2 country code for web search localization.
    pub country: Option<String>,
    /// City name for web search localization.
    pub city: Option<String>,
    /// Region/state for web search localization.
    pub region: Option<String>,
    /// Web search context retrieval depth.
    pub search_context_size: Option<SearchContextSize>,
    /// Whether to enable the code interpreter tool.
    pub code_interpreter: bool,
    /// Developer-level instructions for the model.
    pub instructions: Option<String>,
    /// Whether to store the response (retained for 30 days).
    pub store: Option<bool>,
    /// Key-value metadata.
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Minimal valid queued response JSON.
    fn queued_response_json() -> serde_json::Value {
        serde_json::json!({
            "id": "resp_test123",
            "object": "response",
            "status": "queued",
            "output": [],
            "model": "o3-deep-research-2025-06-26",
            "created_at": 1700000000.0
        })
    }

    /// Completed response JSON with full output (web search, reasoning, message).
    fn completed_response_json() -> serde_json::Value {
        serde_json::json!({
            "id": "resp_completed456",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "completed"
                },
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "Analyzing results"}]
                },
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Here is the report...",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://example.com",
                                    "title": "Example",
                                    "start_index": 0,
                                    "end_index": 10
                                }
                            ]
                        }
                    ]
                }
            ],
            "model": "o3-deep-research-2025-06-26",
            "created_at": 1700000000.0,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 500,
                "total_tokens": 600
            }
        })
    }

    /// Failed response JSON.
    fn failed_response_json() -> serde_json::Value {
        serde_json::json!({
            "id": "resp_failed789",
            "object": "response",
            "status": "failed",
            "output": [],
            "model": "o3-deep-research-2025-06-26",
            "created_at": 1700000000.0,
            "error": {
                "code": "server_error",
                "message": "An internal error occurred"
            }
        })
    }

    // -----------------------------------------------------------------------
    // 1. Client construction
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn client_new_succeeds() {
        let result = DeepResearchClient::new("test-key");
        assert!(
            result.is_ok(),
            "DeepResearchClient::new should succeed with a valid key"
        );
    }

    #[tokio::test]
    async fn client_with_base_url_succeeds() {
        let mock_server = MockServer::start().await;
        let result = DeepResearchClient::with_base_url("test-key", &mock_server.uri());
        assert!(
            result.is_ok(),
            "with_base_url should succeed with a valid URL"
        );
    }

    #[tokio::test]
    async fn client_base_url_trims_trailing_slash() {
        // Start a mock server and register a handler for POST /responses.
        // If the base URL trailing slash is NOT trimmed, the request would go to
        // "//responses" and fail to match the mock, causing the test to fail.
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(queued_response_json()))
            .mount(&mock_server)
            .await;

        // Append a trailing slash to the URI — the client must strip it.
        let base_url_with_slash = format!("{}/", mock_server.uri());
        let client = DeepResearchClient::with_base_url("test-key", &base_url_with_slash)
            .expect("client construction should succeed");

        let result = client
            .create(
                "test query",
                DeepResearchModel::O3,
                CreateOptions::default(),
            )
            .await;

        assert!(
            result.is_ok(),
            "request should reach the mock (trailing slash trimmed): {:?}",
            result.err()
        );
    }

    // -----------------------------------------------------------------------
    // 2. CreateOptions defaults
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn create_options_default() {
        let opts = CreateOptions::default();
        assert!(opts.country.is_none(), "country should default to None");
        assert!(opts.city.is_none(), "city should default to None");
        assert!(opts.region.is_none(), "region should default to None");
        assert!(
            opts.search_context_size.is_none(),
            "search_context_size should default to None"
        );
        assert!(
            !opts.code_interpreter,
            "code_interpreter should default to false"
        );
        assert!(
            opts.instructions.is_none(),
            "instructions should default to None"
        );
        assert!(opts.store.is_none(), "store should default to None");
        assert!(opts.metadata.is_none(), "metadata should default to None");
    }

    // -----------------------------------------------------------------------
    // 3. Request body verification (via wiremock body capture)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn create_sends_correct_request_body() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(queued_response_json()))
            .mount(&mock_server)
            .await;

        let client = DeepResearchClient::with_base_url("test-key", &mock_server.uri())
            .expect("client construction should succeed");

        client
            .create(
                "What is the capital of France?",
                DeepResearchModel::O3,
                CreateOptions::default(),
            )
            .await
            .expect("create should succeed");

        // Retrieve the captured request and inspect its body.
        let received = mock_server.received_requests().await.unwrap();
        assert_eq!(
            received.len(),
            1,
            "exactly one request should have been sent"
        );

        let body: serde_json::Value =
            serde_json::from_slice(&received[0].body).expect("request body should be valid JSON");

        // model field
        assert_eq!(
            body["model"],
            serde_json::json!("o3-deep-research-2025-06-26"),
            "model field should match O3 model string"
        );

        // background must be true
        assert_eq!(
            body["background"],
            serde_json::json!(true),
            "background should be true"
        );

        // reasoning.summary must be "auto"
        assert_eq!(
            body["reasoning"]["summary"],
            serde_json::json!("auto"),
            "reasoning.summary should be 'auto'"
        );

        // input must contain a user message with the query
        let input = body["input"].as_array().expect("input should be an array");
        assert_eq!(
            input.len(),
            1,
            "default options should produce exactly one input message"
        );
        assert_eq!(input[0]["role"], serde_json::json!("user"));
        assert_eq!(
            input[0]["content"],
            serde_json::json!("What is the capital of France?")
        );

        // tools must contain web_search_preview
        let tools = body["tools"].as_array().expect("tools should be an array");
        assert_eq!(
            tools.len(),
            1,
            "default options should produce exactly one tool"
        );
        assert_eq!(tools[0]["type"], serde_json::json!("web_search_preview"));
    }

    #[tokio::test]
    async fn create_with_location_includes_user_location() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(queued_response_json()))
            .mount(&mock_server)
            .await;

        let client = DeepResearchClient::with_base_url("test-key", &mock_server.uri())
            .expect("client construction should succeed");

        let opts = CreateOptions {
            country: Some("US".to_string()),
            city: Some("San Francisco".to_string()),
            region: Some("California".to_string()),
            ..Default::default()
        };

        client
            .create("test query", DeepResearchModel::O3, opts)
            .await
            .expect("create should succeed");

        let received = mock_server.received_requests().await.unwrap();
        let body: serde_json::Value =
            serde_json::from_slice(&received[0].body).expect("request body should be valid JSON");

        let tools = body["tools"].as_array().expect("tools should be an array");
        let web_search = tools
            .iter()
            .find(|t| t["type"] == "web_search_preview")
            .expect("web_search_preview tool should be present");

        let user_location = &web_search["user_location"];
        assert_eq!(
            user_location["type"],
            serde_json::json!("approximate"),
            "user_location.type should be 'approximate'"
        );
        assert_eq!(user_location["country"], serde_json::json!("US"));
        assert_eq!(user_location["city"], serde_json::json!("San Francisco"));
        assert_eq!(user_location["region"], serde_json::json!("California"));
    }

    #[tokio::test]
    async fn create_with_instructions_includes_developer_message() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(queued_response_json()))
            .mount(&mock_server)
            .await;

        let client = DeepResearchClient::with_base_url("test-key", &mock_server.uri())
            .expect("client construction should succeed");

        let opts = CreateOptions {
            instructions: Some("Be concise and cite sources.".to_string()),
            ..Default::default()
        };

        client
            .create("test query", DeepResearchModel::O3, opts)
            .await
            .expect("create should succeed");

        let received = mock_server.received_requests().await.unwrap();
        let body: serde_json::Value =
            serde_json::from_slice(&received[0].body).expect("request body should be valid JSON");

        let input = body["input"].as_array().expect("input should be an array");
        assert_eq!(
            input.len(),
            2,
            "instructions + query should produce two input messages"
        );

        // Developer message must come first.
        assert_eq!(input[0]["role"], serde_json::json!("developer"));
        assert_eq!(
            input[0]["content"],
            serde_json::json!("Be concise and cite sources.")
        );

        // User message must come second.
        assert_eq!(input[1]["role"], serde_json::json!("user"));
        assert_eq!(input[1]["content"], serde_json::json!("test query"));
    }

    #[tokio::test]
    async fn create_with_code_interpreter_includes_tool() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(queued_response_json()))
            .mount(&mock_server)
            .await;

        let client = DeepResearchClient::with_base_url("test-key", &mock_server.uri())
            .expect("client construction should succeed");

        let opts = CreateOptions {
            code_interpreter: true,
            ..Default::default()
        };

        client
            .create("test query", DeepResearchModel::O3, opts)
            .await
            .expect("create should succeed");

        let received = mock_server.received_requests().await.unwrap();
        let body: serde_json::Value =
            serde_json::from_slice(&received[0].body).expect("request body should be valid JSON");

        let tools = body["tools"].as_array().expect("tools should be an array");
        assert_eq!(
            tools.len(),
            2,
            "code_interpreter=true should produce two tools"
        );

        let has_web_search = tools.iter().any(|t| t["type"] == "web_search_preview");
        let has_code_interpreter = tools.iter().any(|t| t["type"] == "code_interpreter");

        assert!(has_web_search, "tools should include web_search_preview");
        assert!(
            has_code_interpreter,
            "tools should include code_interpreter"
        );
    }

    #[tokio::test]
    async fn create_rejects_non_medium_search_context_size_before_request() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(queued_response_json()))
            .mount(&mock_server)
            .await;

        let client = DeepResearchClient::with_base_url("test-key", &mock_server.uri())
            .expect("client construction should succeed");

        let opts = CreateOptions {
            search_context_size: Some(SearchContextSize::Low),
            ..Default::default()
        };

        let err = client
            .create("test query", DeepResearchModel::O3, opts)
            .await
            .expect_err("low search context should be rejected locally");

        assert!(
            err.to_string()
                .contains("only supports search_context_size 'medium'"),
            "error should explain the supported Deep Research search context size: {err}"
        );

        let received = mock_server.received_requests().await.unwrap();
        assert!(
            received.is_empty(),
            "invalid search context size should not send an API request"
        );
    }

    // -----------------------------------------------------------------------
    // 4. Response handling — create
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn create_success_response() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(queued_response_json()))
            .mount(&mock_server)
            .await;

        let client = DeepResearchClient::with_base_url("test-key", &mock_server.uri())
            .expect("client construction should succeed");

        let response = client
            .create(
                "test query",
                DeepResearchModel::O3,
                CreateOptions::default(),
            )
            .await
            .expect("create should succeed with 200 response");

        assert_eq!(response.id, "resp_test123");
        assert_eq!(response.object, "response");
        assert_eq!(response.status, ResponseStatus::Queued);
        assert!(
            response.output.is_empty(),
            "queued response should have empty output"
        );
        assert_eq!(response.model, "o3-deep-research-2025-06-26");
        assert!((response.created_at - 1700000000.0).abs() < f64::EPSILON);
        assert!(response.error.is_none());
        assert!(response.usage.is_none());
    }

    #[tokio::test]
    async fn create_error_response() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(400).set_body_string(
                r#"{"error":{"message":"Invalid request","type":"invalid_request_error"}}"#,
            ))
            .mount(&mock_server)
            .await;

        let client = DeepResearchClient::with_base_url("test-key", &mock_server.uri())
            .expect("client construction should succeed");

        let result = client
            .create(
                "test query",
                DeepResearchModel::O3,
                CreateOptions::default(),
            )
            .await;

        assert!(result.is_err(), "create should return Err on HTTP 400");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("400"),
            "error message should mention the HTTP status code; got: {err_msg}"
        );
    }

    #[tokio::test]
    async fn create_server_error_response() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
            .mount(&mock_server)
            .await;

        let client = DeepResearchClient::with_base_url("test-key", &mock_server.uri())
            .expect("client construction should succeed");

        let result = client
            .create(
                "test query",
                DeepResearchModel::O3,
                CreateOptions::default(),
            )
            .await;

        assert!(result.is_err(), "create should return Err on HTTP 500");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("500"),
            "error message should mention the HTTP status code; got: {err_msg}"
        );
    }

    // -----------------------------------------------------------------------
    // 5. Response handling — retrieve
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn retrieve_success_response() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/responses/resp_completed456"))
            .respond_with(ResponseTemplate::new(200).set_body_json(completed_response_json()))
            .mount(&mock_server)
            .await;

        let client = DeepResearchClient::with_base_url("test-key", &mock_server.uri())
            .expect("client construction should succeed");

        let response = client
            .retrieve("resp_completed456")
            .await
            .expect("retrieve should succeed with 200 response");

        assert_eq!(response.id, "resp_completed456");
        assert_eq!(response.status, ResponseStatus::Completed);
        assert_eq!(
            response.output.len(),
            3,
            "completed response should have 3 output items"
        );

        // Verify usage is present.
        let usage = response
            .usage
            .as_ref()
            .expect("completed response should have usage");
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 500);
        assert_eq!(usage.total_tokens, 600);

        // Verify final_report helper works.
        let report = response
            .final_report()
            .expect("should extract final report text");
        assert_eq!(report, "Here is the report...");

        // Verify citations helper works.
        let citations = response.citations();
        assert_eq!(citations.len(), 1);
        assert_eq!(citations[0].url.as_deref(), Some("https://example.com"));
        assert_eq!(citations[0].title.as_deref(), Some("Example"));

        // Verify research_step_count (web_search + reasoning = 2).
        assert_eq!(response.research_step_count(), 2);

        // Verify is_terminal.
        assert!(response.is_terminal());
    }

    #[tokio::test]
    async fn retrieve_error_response() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/responses/resp_notfound"))
            .respond_with(ResponseTemplate::new(404).set_body_string(
                r#"{"error":{"message":"Response not found","type":"not_found_error"}}"#,
            ))
            .mount(&mock_server)
            .await;

        let client = DeepResearchClient::with_base_url("test-key", &mock_server.uri())
            .expect("client construction should succeed");

        let result = client.retrieve("resp_notfound").await;

        assert!(result.is_err(), "retrieve should return Err on HTTP 404");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("404"),
            "error message should mention the HTTP status code; got: {err_msg}"
        );
    }

    // -----------------------------------------------------------------------
    // 6. Poll behavior
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn poll_returns_on_completed() {
        let mock_server = MockServer::start().await;

        // Return completed status on the very first GET.
        Mock::given(method("GET"))
            .and(path("/responses/resp_completed456"))
            .respond_with(ResponseTemplate::new(200).set_body_json(completed_response_json()))
            .expect(1) // Should only be called once since it's already terminal.
            .mount(&mock_server)
            .await;

        let client = DeepResearchClient::with_base_url("test-key", &mock_server.uri())
            .expect("client construction should succeed");

        let response = client
            .poll_until_complete("resp_completed456")
            .await
            .expect("poll_until_complete should succeed");

        assert_eq!(response.id, "resp_completed456");
        assert_eq!(response.status, ResponseStatus::Completed);
        assert!(response.is_terminal());

        // Verify the mock was called exactly once (no unnecessary polling).
        mock_server.verify().await;
    }

    #[tokio::test]
    async fn poll_returns_on_failed() {
        let mock_server = MockServer::start().await;

        // Return failed status on the very first GET.
        Mock::given(method("GET"))
            .and(path("/responses/resp_failed789"))
            .respond_with(ResponseTemplate::new(200).set_body_json(failed_response_json()))
            .expect(1) // Should only be called once since failed is terminal.
            .mount(&mock_server)
            .await;

        let client = DeepResearchClient::with_base_url("test-key", &mock_server.uri())
            .expect("client construction should succeed");

        let response = client
            .poll_until_complete("resp_failed789")
            .await
            .expect("poll_until_complete should return Ok even for failed responses");

        assert_eq!(response.id, "resp_failed789");
        assert_eq!(response.status, ResponseStatus::Failed);
        assert!(response.is_terminal(), "failed status should be terminal");

        // Verify the error field is populated.
        let error = response
            .error
            .as_ref()
            .expect("failed response should have error details");
        assert_eq!(error.code, "server_error");
        assert!(!error.message.is_empty());

        // Verify the mock was called exactly once.
        mock_server.verify().await;
    }
}
