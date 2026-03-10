//! OpenAI Responses API types for Deep Research.
//!
//! These types model the full request/response shape of the OpenAI Responses API
//! when used with deep research models (o3-deep-research, o4-mini-deep-research).
//! Unlike simpler wrappers, we preserve ALL intermediate output items
//! (web searches, reasoning summaries, MCP tool calls, etc.).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

/// Supported deep research models.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeepResearchModel {
    /// o3-deep-research-2025-06-26
    #[serde(rename = "o3-deep-research-2025-06-26")]
    #[default]
    O3,
    /// o4-mini-deep-research-2025-06-26
    #[serde(rename = "o4-mini-deep-research-2025-06-26")]
    O4Mini,
}

impl DeepResearchModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::O3 => "o3-deep-research-2025-06-26",
            Self::O4Mini => "o4-mini-deep-research-2025-06-26",
        }
    }
}

impl fmt::Display for DeepResearchModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Tool configuration
// ---------------------------------------------------------------------------

/// User location hint for web search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserLocation {
    /// Must be "approximate".
    #[serde(rename = "type")]
    pub location_type: String,
    /// ISO 3166-1 alpha-2 country code (e.g., "US", "GB").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    /// City name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    /// Region/state.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
}

impl Default for UserLocation {
    fn default() -> Self {
        Self::new()
    }
}

impl UserLocation {
    pub fn new() -> Self {
        Self {
            location_type: "approximate".to_string(),
            country: None,
            city: None,
            region: None,
        }
    }

    pub fn with_country(mut self, country: impl Into<String>) -> Self {
        self.country = Some(country.into());
        self
    }

    pub fn with_city(mut self, city: impl Into<String>) -> Self {
        self.city = Some(city.into());
        self
    }

    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Returns true if any location field is set.
    pub fn has_any(&self) -> bool {
        self.country.is_some() || self.city.is_some() || self.region.is_some()
    }
}

/// Web search context retrieval depth.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SearchContextSize {
    Low,
    Medium,
    High,
}

impl fmt::Display for SearchContextSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => f.write_str("low"),
            Self::Medium => f.write_str("medium"),
            Self::High => f.write_str("high"),
        }
    }
}

/// A tool to include in the deep research request.
///
/// Deep research supports: web_search_preview, code_interpreter, file_search, mcp.
/// MVP implements web_search_preview and code_interpreter.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Tool {
    /// Web search with optional location and context size configuration.
    #[serde(rename = "web_search_preview")]
    WebSearchPreview {
        #[serde(skip_serializing_if = "Option::is_none")]
        user_location: Option<UserLocation>,
        #[serde(skip_serializing_if = "Option::is_none")]
        search_context_size: Option<SearchContextSize>,
    },

    /// Code interpreter for running Python code during research.
    #[serde(rename = "code_interpreter")]
    CodeInterpreter {},
    // -- Future: file_search, mcp --
    // /// File search over OpenAI vector stores.
    // #[serde(rename = "file_search")]
    // FileSearch {
    //     vector_store_ids: Vec<String>,
    // },
    //
    // /// Remote MCP server as a data source.
    // #[serde(rename = "mcp")]
    // Mcp {
    //     server_label: String,
    //     server_url: String,
    //     require_approval: String,
    //     #[serde(skip_serializing_if = "Option::is_none")]
    //     allowed_tools: Option<Vec<String>>,
    // },
}

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// An input message for the deep research request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputMessage {
    /// "user" for the research query, "developer" for system instructions.
    pub role: String,
    /// The message content.
    pub content: String,
}

/// Reasoning configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    /// "auto" to include reasoning summaries in the output.
    pub summary: String,
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        Self {
            summary: "auto".to_string(),
        }
    }
}

/// Request body for creating a deep research job.
///
/// Sent as POST to /v1/responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateRequest {
    /// The deep research model to use.
    pub model: String,
    /// Input messages (user query + optional developer instructions).
    pub input: Vec<InputMessage>,
    /// Tools available to the model.
    pub tools: Vec<Tool>,
    /// Reasoning configuration.
    pub reasoning: ReasoningConfig,
    /// Run in background mode (recommended for deep research).
    pub background: bool,
    /// Optional: top-level instructions for the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Optional: whether to store the response (retained for 30 days).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Optional: key-value metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// Status of a deep research response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    Queued,
    InProgress,
    Completed,
    Failed,
    Incomplete,
}

impl fmt::Display for ResponseStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Queued => f.write_str("queued"),
            Self::InProgress => f.write_str("in_progress"),
            Self::Completed => f.write_str("completed"),
            Self::Failed => f.write_str("failed"),
            Self::Incomplete => f.write_str("incomplete"),
        }
    }
}

/// API-level error information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseError {
    pub code: String,
    pub message: String,
}

/// Token usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
}

/// The top-level response object from the Responses API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    /// Unique response ID (starts with "resp_").
    pub id: String,
    /// Always "response".
    pub object: String,
    /// Current status of the research job.
    pub status: ResponseStatus,
    /// Full output array — includes ALL intermediate items.
    #[serde(default)]
    pub output: Vec<OutputItem>,
    /// Error details (if status is "failed").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ResponseError>,
    /// Token usage (available when completed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Model used.
    #[serde(default)]
    pub model: String,
    /// Creation timestamp (Unix seconds).
    #[serde(default)]
    pub created_at: f64,
    /// Metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

// ---------------------------------------------------------------------------
// Output items — the FULL intermediate output
// ---------------------------------------------------------------------------

/// A single item in the response output array.
///
/// Deep research produces a rich stream of intermediate items showing
/// what the model searched, reasoned about, and generated. Unlike simpler
/// wrappers that only return the final message, we preserve everything.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OutputItem {
    /// The final (or intermediate) assistant message with the research report.
    #[serde(rename = "message")]
    Message {
        id: String,
        role: String,
        #[serde(default)]
        status: Option<String>,
        #[serde(default)]
        content: Vec<ContentItem>,
    },

    /// A web search call made during research.
    #[serde(rename = "web_search_call")]
    WebSearchCall {
        id: String,
        #[serde(default)]
        status: Option<String>,
    },

    /// Reasoning step with a summary of the model's thinking.
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
        #[serde(default)]
        summary: Vec<SummaryItem>,
    },

    /// An MCP tool call made to a connected remote MCP server.
    #[serde(rename = "mcp_tool_call")]
    McpToolCall {
        id: String,
        #[serde(default)]
        server_label: Option<String>,
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        arguments: Option<String>,
        #[serde(default)]
        output: Option<String>,
        #[serde(default)]
        error: Option<String>,
    },

    /// A code interpreter invocation.
    #[serde(rename = "code_interpreter_call")]
    CodeInterpreterCall {
        id: String,
        #[serde(default)]
        status: Option<String>,
        #[serde(default)]
        code: Option<String>,
        #[serde(default)]
        results: Option<Vec<serde_json::Value>>,
    },

    /// A file search call over vector stores.
    #[serde(rename = "file_search_call")]
    FileSearchCall {
        id: String,
        #[serde(default)]
        status: Option<String>,
        #[serde(default)]
        results: Option<Vec<serde_json::Value>>,
    },

    /// Catch-all for unknown output item types (forward compatibility).
    #[serde(other)]
    Unknown,
}

/// Content within a message output item.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentItem {
    /// Text content with optional URL citation annotations.
    #[serde(rename = "output_text")]
    OutputText {
        text: String,
        #[serde(default)]
        annotations: Vec<Annotation>,
    },

    /// Catch-all for unknown content types.
    #[serde(other)]
    Unknown,
}

/// A URL citation annotation on text content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    /// Annotation type (e.g., "url_citation").
    #[serde(rename = "type")]
    pub annotation_type: String,
    /// The cited URL.
    #[serde(default)]
    pub url: Option<String>,
    /// Title of the cited page.
    #[serde(default)]
    pub title: Option<String>,
    /// Start character index in the parent text.
    #[serde(default)]
    pub start_index: Option<usize>,
    /// End character index in the parent text.
    #[serde(default)]
    pub end_index: Option<usize>,
}

/// A summary text item within a reasoning output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryItem {
    /// Always "summary_text".
    #[serde(rename = "type")]
    pub summary_type: String,
    /// The reasoning summary text.
    pub text: String,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

impl Response {
    /// Extract the final assistant message text (the research report).
    pub fn final_report(&self) -> Option<String> {
        // Walk output in reverse to find the last message with text content.
        for item in self.output.iter().rev() {
            if let OutputItem::Message { content, .. } = item {
                for c in content {
                    if let ContentItem::OutputText { text, .. } = c {
                        if !text.is_empty() {
                            return Some(text.clone());
                        }
                    }
                }
            }
        }
        None
    }

    /// Extract all URL citations from the final message.
    pub fn citations(&self) -> Vec<&Annotation> {
        let mut citations = Vec::new();
        for item in self.output.iter().rev() {
            if let OutputItem::Message { content, .. } = item {
                for c in content {
                    if let ContentItem::OutputText { annotations, .. } = c {
                        for ann in annotations {
                            if ann.annotation_type == "url_citation" {
                                citations.push(ann);
                            }
                        }
                    }
                }
                if !citations.is_empty() {
                    break; // Only citations from the last message
                }
            }
        }
        citations
    }

    /// Count intermediate research steps (web searches, reasoning, tool calls).
    pub fn research_step_count(&self) -> usize {
        self.output
            .iter()
            .filter(|item| {
                matches!(
                    item,
                    OutputItem::WebSearchCall { .. }
                        | OutputItem::Reasoning { .. }
                        | OutputItem::McpToolCall { .. }
                        | OutputItem::CodeInterpreterCall { .. }
                        | OutputItem::FileSearchCall { .. }
                )
            })
            .count()
    }

    /// Extract all reasoning summaries.
    pub fn reasoning_summaries(&self) -> Vec<&str> {
        self.output
            .iter()
            .filter_map(|item| {
                if let OutputItem::Reasoning { summary, .. } = item {
                    Some(summary.iter().map(|s| s.text.as_str()))
                } else {
                    None
                }
            })
            .flatten()
            .collect()
    }

    /// Check if the response is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            ResponseStatus::Completed | ResponseStatus::Failed | ResponseStatus::Incomplete
        )
    }
}
