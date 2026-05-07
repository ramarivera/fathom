//! MCP server exposing deep research tools.
//!
//! Provides three tools matching the fbettag reference pattern:
//! - `create_research`  — start a new deep research job
//! - `check_status`     — poll job status and progress
//! - `get_results`      — retrieve full results with intermediate output
//!
//! Unlike fbettag's implementation, we preserve ALL intermediate output items
//! (web searches, reasoning summaries, code interpreter calls, etc.) and
//! support web search configuration (user location, context size).

use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, Content, ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler,
};
use serde::Deserialize;

use crate::api::client::{CreateOptions, DeepResearchClient};
use crate::api::types::{DeepResearchModel, SearchContextSize};

// ---------------------------------------------------------------------------
// Tool parameter types
// ---------------------------------------------------------------------------

/// Parameters for creating a new deep research job.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CreateResearchParams {
    /// The research query or question to investigate.
    #[schemars(description = "The research query or question to investigate")]
    pub query: String,

    /// Model to use: "o3" (default, full power) or "o4-mini" (faster, cheaper).
    #[schemars(description = "Model: 'o3' (default, full power) or 'o4-mini' (faster, cheaper)")]
    pub model: Option<String>,

    /// ISO 3166-1 alpha-2 country code for localizing web search results (e.g., "US", "GB").
    #[schemars(description = "ISO 3166-1 alpha-2 country code for web search localization")]
    pub country: Option<String>,

    /// City name for localizing web search results.
    #[schemars(description = "City name for web search localization")]
    pub city: Option<String>,

    /// Region/state for localizing web search results.
    #[schemars(description = "Region/state for web search localization")]
    pub region: Option<String>,

    /// Web search context depth. Deep Research currently supports only "medium".
    #[schemars(
        description = "Web search context depth. Deep Research currently supports only 'medium'; omit this field unless you need to pass 'medium' explicitly."
    )]
    pub search_context_size: Option<String>,

    /// Whether to enable the code interpreter tool during research.
    #[schemars(description = "Enable code interpreter during research (default: false)")]
    pub code_interpreter: Option<bool>,

    /// Optional developer-level instructions for the model.
    #[schemars(description = "Optional developer-level instructions for the model")]
    pub instructions: Option<String>,
}

/// Parameters for checking research job status.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CheckStatusParams {
    /// The response ID returned by create_research (starts with "resp_").
    #[schemars(description = "Response ID from create_research (starts with 'resp_')")]
    pub response_id: String,
}

/// Parameters for retrieving full research results.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetResultsParams {
    /// The response ID returned by create_research (starts with "resp_").
    #[schemars(description = "Response ID from create_research (starts with 'resp_')")]
    pub response_id: String,

    /// Whether to include intermediate research steps (web searches, reasoning, etc.).
    /// Defaults to true — this is fathom's key differentiator.
    #[schemars(
        description = "Include intermediate steps (web searches, reasoning). Default: true"
    )]
    pub include_steps: Option<bool>,
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

/// The fathom MCP server.
#[derive(Debug, Clone)]
pub struct FathomServer {
    client: DeepResearchClient,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl FathomServer {
    /// Create a new FathomServer with the given OpenAI client.
    pub fn new(client: DeepResearchClient) -> Self {
        Self {
            client,
            tool_router: Self::tool_router(),
        }
    }

    /// Start a new deep research job.
    ///
    /// Returns the response ID for polling. Deep research jobs typically
    /// take 5-30 minutes. Use check_status to monitor progress and
    /// get_results to retrieve the full report when complete.
    #[tool(
        name = "create_research",
        description = "Start a new deep research job. Returns a response ID for polling. Jobs take 5-30 minutes. Use check_status to monitor and get_results when complete."
    )]
    async fn create_research(
        &self,
        Parameters(params): Parameters<CreateResearchParams>,
    ) -> Result<CallToolResult, McpError> {
        let model = match params.model.as_deref() {
            Some("o4-mini") | Some("o4_mini") => DeepResearchModel::O4Mini,
            _ => DeepResearchModel::O3,
        };

        let search_context_size = params.search_context_size.as_deref().and_then(|s| match s {
            "medium" => Some(SearchContextSize::Medium),
            _ => None,
        });

        let options = CreateOptions {
            country: params.country,
            city: params.city,
            region: params.region,
            search_context_size,
            code_interpreter: params.code_interpreter.unwrap_or(false),
            instructions: params.instructions,
            store: Some(true),
            metadata: None,
        };

        let response = self
            .client
            .create(&params.query, model, options)
            .await
            .map_err(|e| {
                McpError::internal_error(format!("Failed to create research: {e}"), None)
            })?;

        let result = serde_json::json!({
            "response_id": response.id,
            "status": response.status.to_string(),
            "model": response.model,
            "message": "Research job created. Use check_status to monitor progress."
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&result).unwrap_or_default(),
        )]))
    }

    /// Check the current status and progress of a research job.
    ///
    /// Returns the status, number of research steps completed so far,
    /// and reasoning summaries if available.
    #[tool(
        name = "check_status",
        description = "Check deep research job status and progress. Returns status, step count, and reasoning summaries."
    )]
    async fn check_status(
        &self,
        Parameters(params): Parameters<CheckStatusParams>,
    ) -> Result<CallToolResult, McpError> {
        let response = self
            .client
            .retrieve(&params.response_id)
            .await
            .map_err(|e| McpError::internal_error(format!("Failed to check status: {e}"), None))?;

        let summaries = response.reasoning_summaries();

        let mut result = serde_json::json!({
            "response_id": response.id,
            "status": response.status.to_string(),
            "research_steps": response.research_step_count(),
        });

        if !summaries.is_empty() {
            result["reasoning_summaries"] = serde_json::json!(summaries);
        }

        if let Some(error) = &response.error {
            result["error"] = serde_json::json!({
                "code": error.code,
                "message": error.message,
            });
        }

        if let Some(usage) = &response.usage {
            result["usage"] = serde_json::json!({
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            });
        }

        let is_done = response.is_terminal();
        result["is_terminal"] = serde_json::json!(is_done);
        if !is_done {
            result["hint"] = serde_json::json!("Job still running. Poll again in 10-30 seconds.");
        }

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&result).unwrap_or_default(),
        )]))
    }

    /// Retrieve the full results of a completed research job.
    ///
    /// Returns the research report, citations, and optionally all
    /// intermediate steps (web searches, reasoning, code interpreter calls).
    #[tool(
        name = "get_results",
        description = "Get full research results including report, citations, and intermediate steps (web searches, reasoning, code runs)."
    )]
    async fn get_results(
        &self,
        Parameters(params): Parameters<GetResultsParams>,
    ) -> Result<CallToolResult, McpError> {
        let response = self
            .client
            .retrieve(&params.response_id)
            .await
            .map_err(|e| McpError::internal_error(format!("Failed to get results: {e}"), None))?;

        let include_steps = params.include_steps.unwrap_or(true);

        let mut result = serde_json::json!({
            "response_id": response.id,
            "status": response.status.to_string(),
            "model": response.model,
        });

        // Add the final report if available.
        if let Some(report) = response.final_report() {
            result["report"] = serde_json::json!(report);
        }

        // Add citations.
        let citations: Vec<_> = response
            .citations()
            .iter()
            .filter_map(|ann| {
                ann.url.as_ref().map(|url| {
                    serde_json::json!({
                        "url": url,
                        "title": ann.title,
                    })
                })
            })
            .collect();

        if !citations.is_empty() {
            result["citations"] = serde_json::json!(citations);
        }

        // Add intermediate steps if requested (fathom's differentiator).
        if include_steps {
            result["research_steps"] = serde_json::json!(response.research_step_count());

            let summaries = response.reasoning_summaries();
            if !summaries.is_empty() {
                result["reasoning_summaries"] = serde_json::json!(summaries);
            }

            // Include full output array as JSON for maximum fidelity.
            result["output"] = serde_json::to_value(&response.output).unwrap_or_default();
        }

        // Add usage stats.
        if let Some(usage) = &response.usage {
            result["usage"] = serde_json::json!({
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            });
        }

        // Add error details.
        if let Some(error) = &response.error {
            result["error"] = serde_json::json!({
                "code": error.code,
                "message": error.message,
            });
        }

        // Check if job is still running.
        if !response.is_terminal() {
            result["warning"] =
                serde_json::json!("Research is still in progress. Results may be incomplete.");
        }

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&result).unwrap_or_default(),
        )]))
    }
}

#[tool_handler]
impl ServerHandler for FathomServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "Fathom: Deep Research API wrapper. Use create_research to start a job, \
                 check_status to monitor progress, and get_results to retrieve the full \
                 report with citations and intermediate research steps."
                .to_string(),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::ServerHandler;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Minimal queued response JSON (POST /responses returns this).
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

    /// In-progress response JSON with one web search step and one reasoning step.
    fn in_progress_response_json() -> serde_json::Value {
        serde_json::json!({
            "id": "resp_test123",
            "object": "response",
            "status": "in_progress",
            "output": [
                {"type": "web_search_call", "id": "ws_1", "status": "completed"},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "Found relevant info"}]
                }
            ],
            "model": "o3-deep-research-2025-06-26",
            "created_at": 1700000000.0
        })
    }

    /// Completed response JSON with a full report and usage stats.
    fn completed_response_json() -> serde_json::Value {
        serde_json::json!({
            "id": "resp_test123",
            "object": "response",
            "status": "completed",
            "output": [
                {"type": "web_search_call", "id": "ws_1", "status": "completed"},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "Found relevant info"}]
                },
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Research report here",
                            "annotations": []
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

    // -----------------------------------------------------------------------
    // Server construction and info
    // -----------------------------------------------------------------------

    /// 1. Verify that constructing a FathomServer with a dummy key does not panic.
    #[test]
    fn server_construction() {
        let client = DeepResearchClient::with_base_url("test-key", "http://localhost:9999")
            .expect("client construction should not fail");
        let _server = FathomServer::new(client);
        // If we reach here without panicking, the test passes.
    }

    /// 2. Verify that get_info() returns a ServerInfo with tools capability enabled.
    #[test]
    fn server_info_has_tools_capability() {
        let client = DeepResearchClient::with_base_url("test-key", "http://localhost:9999")
            .expect("client construction should not fail");
        let server = FathomServer::new(client);

        let info = server.get_info();

        // ServerCapabilities is a plain struct (not Option); tools within it is Option.
        assert!(
            info.capabilities.tools.is_some(),
            "tools capability should be enabled in ServerInfo"
        );
    }

    /// 3. Verify that get_info() includes instruction text.
    #[test]
    fn server_info_has_instructions() {
        let client = DeepResearchClient::with_base_url("test-key", "http://localhost:9999")
            .expect("client construction should not fail");
        let server = FathomServer::new(client);

        let info = server.get_info();

        let instructions = info
            .instructions
            .expect("ServerInfo should have instructions");
        assert!(!instructions.is_empty(), "instructions should not be empty");
        // Spot-check that the instructions mention the key tools.
        assert!(
            instructions.contains("create_research"),
            "instructions should mention create_research"
        );
        assert!(
            instructions.contains("check_status"),
            "instructions should mention check_status"
        );
        assert!(
            instructions.contains("get_results"),
            "instructions should mention get_results"
        );
    }

    // -----------------------------------------------------------------------
    // Tool parameter deserialization
    // -----------------------------------------------------------------------

    /// 4. Deserialize minimal CreateResearchParams (only query).
    #[test]
    fn create_research_params_minimal() {
        let json = r#"{"query": "test"}"#;
        let params: CreateResearchParams =
            serde_json::from_str(json).expect("should deserialize minimal params");

        assert_eq!(params.query, "test");
        assert!(params.model.is_none());
        assert!(params.country.is_none());
        assert!(params.city.is_none());
        assert!(params.region.is_none());
        assert!(params.search_context_size.is_none());
        assert!(params.code_interpreter.is_none());
        assert!(params.instructions.is_none());
    }

    /// 5. Deserialize CreateResearchParams with all fields populated.
    #[test]
    fn create_research_params_full() {
        let json = serde_json::json!({
            "query": "What is the future of AI?",
            "model": "o4-mini",
            "country": "US",
            "city": "San Francisco",
            "region": "California",
            "search_context_size": "medium",
            "code_interpreter": true,
            "instructions": "Be concise."
        })
        .to_string();

        let params: CreateResearchParams =
            serde_json::from_str(&json).expect("should deserialize full params");

        assert_eq!(params.query, "What is the future of AI?");
        assert_eq!(params.model.as_deref(), Some("o4-mini"));
        assert_eq!(params.country.as_deref(), Some("US"));
        assert_eq!(params.city.as_deref(), Some("San Francisco"));
        assert_eq!(params.region.as_deref(), Some("California"));
        assert_eq!(params.search_context_size.as_deref(), Some("medium"));
        assert_eq!(params.code_interpreter, Some(true));
        assert_eq!(params.instructions.as_deref(), Some("Be concise."));
    }

    /// 6. Deserialize CheckStatusParams.
    #[test]
    fn check_status_params() {
        let json = r#"{"response_id": "resp_test"}"#;
        let params: CheckStatusParams =
            serde_json::from_str(json).expect("should deserialize CheckStatusParams");

        assert_eq!(params.response_id, "resp_test");
    }

    /// 7. Deserialize minimal GetResultsParams (include_steps defaults to None).
    #[test]
    fn get_results_params_minimal() {
        let json = r#"{"response_id": "resp_test"}"#;
        let params: GetResultsParams =
            serde_json::from_str(json).expect("should deserialize minimal GetResultsParams");

        assert_eq!(params.response_id, "resp_test");
        assert!(
            params.include_steps.is_none(),
            "include_steps should default to None when not provided"
        );
    }

    /// 8. Deserialize GetResultsParams with include_steps explicitly set to false.
    #[test]
    fn get_results_params_full() {
        let json = r#"{"response_id": "resp_test", "include_steps": false}"#;
        let params: GetResultsParams =
            serde_json::from_str(json).expect("should deserialize full GetResultsParams");

        assert_eq!(params.response_id, "resp_test");
        assert_eq!(params.include_steps, Some(false));
    }

    // -----------------------------------------------------------------------
    // MCP tool integration (wiremock-backed HTTP)
    // -----------------------------------------------------------------------

    /// 9. create_research tool — mock POST /responses, verify response_id in result.
    #[tokio::test]
    async fn tool_create_research_success() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(queued_response_json()))
            .mount(&mock_server)
            .await;

        let client = DeepResearchClient::with_base_url("test-key", &mock_server.uri()).unwrap();
        let server = FathomServer::new(client);

        let params = CreateResearchParams {
            query: "What is quantum computing?".to_string(),
            model: None,
            country: None,
            city: None,
            region: None,
            search_context_size: None,
            code_interpreter: None,
            instructions: None,
        };

        let result = server
            .create_research(Parameters(params))
            .await
            .expect("create_research should succeed");

        // The result must be a success (not an error).
        assert!(
            !result.is_error.unwrap_or(false),
            "result should not be an error"
        );

        // Extract the text content and verify it contains the response_id.
        let text = result
            .content
            .iter()
            .find_map(|c| {
                if let rmcp::model::RawContent::Text(t) = &c.raw {
                    Some(t.text.clone())
                } else {
                    None
                }
            })
            .expect("result should contain text content");

        assert!(
            text.contains("resp_test123"),
            "result text should contain the response_id; got: {text}"
        );
        assert!(
            text.contains("queued"),
            "result text should contain the status; got: {text}"
        );
    }

    /// 10. check_status tool — mock GET /responses/{id} returning in_progress,
    ///     verify result contains status and research_steps.
    #[tokio::test]
    async fn tool_check_status_success() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/responses/resp_test123"))
            .respond_with(ResponseTemplate::new(200).set_body_json(in_progress_response_json()))
            .mount(&mock_server)
            .await;

        let client = DeepResearchClient::with_base_url("test-key", &mock_server.uri()).unwrap();
        let server = FathomServer::new(client);

        let params = CheckStatusParams {
            response_id: "resp_test123".to_string(),
        };

        let result = server
            .check_status(Parameters(params))
            .await
            .expect("check_status should succeed");

        assert!(
            !result.is_error.unwrap_or(false),
            "result should not be an error"
        );

        let text = result
            .content
            .iter()
            .find_map(|c| {
                if let rmcp::model::RawContent::Text(t) = &c.raw {
                    Some(t.text.clone())
                } else {
                    None
                }
            })
            .expect("result should contain text content");

        assert!(
            text.contains("in_progress"),
            "result text should contain the status; got: {text}"
        );
        assert!(
            text.contains("research_steps"),
            "result text should contain research_steps; got: {text}"
        );
        // The in-progress fixture has 2 steps (web_search_call + reasoning).
        assert!(
            text.contains("resp_test123"),
            "result text should contain the response_id; got: {text}"
        );
    }

    /// 11. get_results tool — mock GET /responses/{id} returning completed response
    ///     with a report, verify result contains the report text.
    #[tokio::test]
    async fn tool_get_results_success() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/responses/resp_test123"))
            .respond_with(ResponseTemplate::new(200).set_body_json(completed_response_json()))
            .mount(&mock_server)
            .await;

        let client = DeepResearchClient::with_base_url("test-key", &mock_server.uri()).unwrap();
        let server = FathomServer::new(client);

        let params = GetResultsParams {
            response_id: "resp_test123".to_string(),
            include_steps: None, // defaults to true inside get_results
        };

        let result = server
            .get_results(Parameters(params))
            .await
            .expect("get_results should succeed");

        assert!(
            !result.is_error.unwrap_or(false),
            "result should not be an error"
        );

        let text = result
            .content
            .iter()
            .find_map(|c| {
                if let rmcp::model::RawContent::Text(t) = &c.raw {
                    Some(t.text.clone())
                } else {
                    None
                }
            })
            .expect("result should contain text content");

        assert!(
            text.contains("Research report here"),
            "result text should contain the report; got: {text}"
        );
        assert!(
            text.contains("completed"),
            "result text should contain the status; got: {text}"
        );
        assert!(
            text.contains("resp_test123"),
            "result text should contain the response_id; got: {text}"
        );
        // Usage stats should be present since the fixture includes them.
        assert!(
            text.contains("input_tokens"),
            "result text should contain usage stats; got: {text}"
        );
    }
}
