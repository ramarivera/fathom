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
    ErrorData as McpError, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, Content, ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
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

    /// Web search context depth: "low", "medium", or "high".
    #[schemars(description = "Web search context depth: 'low', 'medium', or 'high'")]
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
            "low" => Some(SearchContextSize::Low),
            "medium" => Some(SearchContextSize::Medium),
            "high" => Some(SearchContextSize::High),
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
            .map_err(|e| McpError::internal_error(format!("Failed to create research: {e}"), None))?;

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
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(
                "Fathom: Deep Research API wrapper. Use create_research to start a job, \
                 check_status to monitor progress, and get_results to retrieve the full \
                 report with citations and intermediate research steps."
                    .to_string(),
            )
    }
}
