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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{from_str, from_value, json, to_string, to_value};

    // -----------------------------------------------------------------------
    // DeepResearchModel
    // -----------------------------------------------------------------------

    #[test]
    fn serde_model_o3_roundtrip() {
        let model = DeepResearchModel::O3;
        let json_str = to_string(&model).expect("serialize O3");
        assert_eq!(json_str, r#""o3-deep-research-2025-06-26""#);
        let back: DeepResearchModel = from_str(&json_str).expect("deserialize O3");
        assert_eq!(back, DeepResearchModel::O3);
    }

    #[test]
    fn serde_model_o4mini_roundtrip() {
        let model = DeepResearchModel::O4Mini;
        let json_str = to_string(&model).expect("serialize O4Mini");
        assert_eq!(json_str, r#""o4-mini-deep-research-2025-06-26""#);
        let back: DeepResearchModel = from_str(&json_str).expect("deserialize O4Mini");
        assert_eq!(back, DeepResearchModel::O4Mini);
    }

    #[test]
    fn model_display() {
        assert_eq!(
            DeepResearchModel::O3.to_string(),
            "o3-deep-research-2025-06-26"
        );
        assert_eq!(
            DeepResearchModel::O4Mini.to_string(),
            "o4-mini-deep-research-2025-06-26"
        );
    }

    #[test]
    fn model_default_is_o3() {
        let model = DeepResearchModel::default();
        assert_eq!(model, DeepResearchModel::O3);
    }

    #[test]
    fn model_as_str() {
        assert_eq!(
            DeepResearchModel::O3.as_str(),
            "o3-deep-research-2025-06-26"
        );
        assert_eq!(
            DeepResearchModel::O4Mini.as_str(),
            "o4-mini-deep-research-2025-06-26"
        );
    }

    // -----------------------------------------------------------------------
    // UserLocation
    // -----------------------------------------------------------------------

    #[test]
    fn user_location_new_defaults() {
        let loc = UserLocation::new();
        assert_eq!(loc.location_type, "approximate");
        assert!(loc.country.is_none());
        assert!(loc.city.is_none());
        assert!(loc.region.is_none());
    }

    #[test]
    fn user_location_builder_chain() {
        let loc = UserLocation::new()
            .with_country("US")
            .with_city("San Francisco")
            .with_region("California");
        assert_eq!(loc.country.as_deref(), Some("US"));
        assert_eq!(loc.city.as_deref(), Some("San Francisco"));
        assert_eq!(loc.region.as_deref(), Some("California"));
        assert_eq!(loc.location_type, "approximate");
    }

    #[test]
    fn user_location_has_any_empty() {
        let loc = UserLocation::new();
        assert!(!loc.has_any());
    }

    #[test]
    fn user_location_has_any_country() {
        let loc = UserLocation::new().with_country("GB");
        assert!(loc.has_any());
    }

    #[test]
    fn user_location_default_matches_new() {
        let default_loc = UserLocation::default();
        let new_loc = UserLocation::new();
        // Compare field by field since UserLocation doesn't derive PartialEq
        assert_eq!(default_loc.location_type, new_loc.location_type);
        assert_eq!(default_loc.country, new_loc.country);
        assert_eq!(default_loc.city, new_loc.city);
        assert_eq!(default_loc.region, new_loc.region);
    }

    #[test]
    fn user_location_serde_skips_none() {
        let loc = UserLocation::new();
        let val = to_value(&loc).expect("serialize empty UserLocation");
        assert!(val.get("country").is_none(), "country key should be absent");
        assert!(val.get("city").is_none(), "city key should be absent");
        assert!(val.get("region").is_none(), "region key should be absent");
        assert_eq!(val["type"], "approximate");
    }

    #[test]
    fn user_location_serde_includes_values() {
        let loc = UserLocation::new()
            .with_country("DE")
            .with_city("Berlin")
            .with_region("Brandenburg");
        let val = to_value(&loc).expect("serialize full UserLocation");
        assert_eq!(val["type"], "approximate");
        assert_eq!(val["country"], "DE");
        assert_eq!(val["city"], "Berlin");
        assert_eq!(val["region"], "Brandenburg");
    }

    // -----------------------------------------------------------------------
    // SearchContextSize
    // -----------------------------------------------------------------------

    #[test]
    fn search_context_size_serde() {
        let cases = [
            (SearchContextSize::Low, "\"low\""),
            (SearchContextSize::Medium, "\"medium\""),
            (SearchContextSize::High, "\"high\""),
        ];
        for (variant, expected_json) in &cases {
            let serialized = to_string(variant).expect("serialize SearchContextSize");
            assert_eq!(&serialized, expected_json);
            let back: SearchContextSize =
                from_str(&serialized).expect("deserialize SearchContextSize");
            assert_eq!(&back, variant);
        }
    }

    #[test]
    fn search_context_size_display() {
        assert_eq!(SearchContextSize::Low.to_string(), "low");
        assert_eq!(SearchContextSize::Medium.to_string(), "medium");
        assert_eq!(SearchContextSize::High.to_string(), "high");
    }

    // -----------------------------------------------------------------------
    // Tool
    // -----------------------------------------------------------------------

    #[test]
    fn tool_web_search_minimal_serde() {
        let tool = Tool::WebSearchPreview {
            user_location: None,
            search_context_size: None,
        };
        let val = to_value(&tool).expect("serialize minimal WebSearchPreview");
        assert_eq!(val["type"], "web_search_preview");
        assert!(
            val.get("user_location").is_none(),
            "user_location should be absent"
        );
        assert!(
            val.get("search_context_size").is_none(),
            "search_context_size should be absent"
        );
    }

    #[test]
    fn tool_web_search_full_serde() {
        let tool = Tool::WebSearchPreview {
            user_location: Some(UserLocation::new().with_country("JP").with_city("Tokyo")),
            search_context_size: Some(SearchContextSize::High),
        };
        let val = to_value(&tool).expect("serialize full WebSearchPreview");
        assert_eq!(val["type"], "web_search_preview");
        assert_eq!(val["search_context_size"], "high");
        assert_eq!(val["user_location"]["type"], "approximate");
        assert_eq!(val["user_location"]["country"], "JP");
        assert_eq!(val["user_location"]["city"], "Tokyo");

        // Round-trip: deserialize back and re-serialize to confirm stability
        let json_str = to_string(&tool).expect("serialize to string");
        let back: Tool = from_str(&json_str).expect("deserialize full WebSearchPreview");
        let val2 = to_value(&back).expect("re-serialize");
        assert_eq!(val, val2);
    }

    #[test]
    fn tool_code_interpreter_serde() {
        let tool = Tool::CodeInterpreter {};
        let val = to_value(&tool).expect("serialize CodeInterpreter");
        assert_eq!(val["type"], "code_interpreter");

        let json_str = to_string(&tool).expect("serialize to string");
        let back: Tool = from_str(&json_str).expect("deserialize CodeInterpreter");
        let val2 = to_value(&back).expect("re-serialize");
        assert_eq!(val, val2);
    }

    // -----------------------------------------------------------------------
    // ResponseStatus
    // -----------------------------------------------------------------------

    #[test]
    fn response_status_serde_all() {
        let cases = [
            (ResponseStatus::Queued, "\"queued\""),
            (ResponseStatus::InProgress, "\"in_progress\""),
            (ResponseStatus::Completed, "\"completed\""),
            (ResponseStatus::Failed, "\"failed\""),
            (ResponseStatus::Incomplete, "\"incomplete\""),
        ];
        for (variant, expected_json) in &cases {
            let serialized = to_string(variant).expect("serialize ResponseStatus");
            assert_eq!(&serialized, expected_json);
            let back: ResponseStatus = from_str(&serialized).expect("deserialize ResponseStatus");
            assert_eq!(&back, variant);
        }
    }

    #[test]
    fn response_status_display() {
        assert_eq!(ResponseStatus::Queued.to_string(), "queued");
        assert_eq!(ResponseStatus::InProgress.to_string(), "in_progress");
        assert_eq!(ResponseStatus::Completed.to_string(), "completed");
        assert_eq!(ResponseStatus::Failed.to_string(), "failed");
        assert_eq!(ResponseStatus::Incomplete.to_string(), "incomplete");
    }

    // -----------------------------------------------------------------------
    // Response helpers — helper to build a minimal valid Response
    // -----------------------------------------------------------------------

    fn make_response(status: ResponseStatus, output: Vec<OutputItem>) -> Response {
        Response {
            id: "resp_test123".to_string(),
            object: "response".to_string(),
            status,
            output,
            error: None,
            usage: None,
            model: "o3-deep-research-2025-06-26".to_string(),
            created_at: 1_700_000_000.0,
            metadata: None,
        }
    }

    #[test]
    fn final_report_from_completed() {
        let response = make_response(
            ResponseStatus::Completed,
            vec![OutputItem::Message {
                id: "msg_001".to_string(),
                role: "assistant".to_string(),
                status: Some("completed".to_string()),
                content: vec![ContentItem::OutputText {
                    text: "The research report text.".to_string(),
                    annotations: vec![],
                }],
            }],
        );
        let report = response.final_report();
        assert_eq!(report.as_deref(), Some("The research report text."));
    }

    #[test]
    fn final_report_empty_output() {
        let response = make_response(ResponseStatus::Completed, vec![]);
        assert!(response.final_report().is_none());
    }

    #[test]
    fn final_report_no_text() {
        let response = make_response(
            ResponseStatus::Completed,
            vec![OutputItem::Message {
                id: "msg_002".to_string(),
                role: "assistant".to_string(),
                status: None,
                content: vec![],
            }],
        );
        assert!(response.final_report().is_none());
    }

    #[test]
    fn citations_extracted() {
        let ann1 = Annotation {
            annotation_type: "url_citation".to_string(),
            url: Some("https://example.com".to_string()),
            title: Some("Example".to_string()),
            start_index: Some(0),
            end_index: Some(10),
        };
        let ann2 = Annotation {
            annotation_type: "url_citation".to_string(),
            url: Some("https://other.org".to_string()),
            title: None,
            start_index: None,
            end_index: None,
        };
        let response = make_response(
            ResponseStatus::Completed,
            vec![OutputItem::Message {
                id: "msg_003".to_string(),
                role: "assistant".to_string(),
                status: Some("completed".to_string()),
                content: vec![ContentItem::OutputText {
                    text: "Some text with citations.".to_string(),
                    annotations: vec![ann1, ann2],
                }],
            }],
        );
        let citations = response.citations();
        assert_eq!(citations.len(), 2);
        assert_eq!(citations[0].url.as_deref(), Some("https://example.com"));
        assert_eq!(citations[1].url.as_deref(), Some("https://other.org"));
    }

    #[test]
    fn citations_empty() {
        let response = make_response(
            ResponseStatus::Completed,
            vec![OutputItem::Message {
                id: "msg_004".to_string(),
                role: "assistant".to_string(),
                status: None,
                content: vec![ContentItem::OutputText {
                    text: "No citations here.".to_string(),
                    annotations: vec![],
                }],
            }],
        );
        assert!(response.citations().is_empty());
    }

    #[test]
    fn research_step_count_mixed() {
        let response = make_response(
            ResponseStatus::Completed,
            vec![
                OutputItem::WebSearchCall {
                    id: "ws_001".to_string(),
                    status: Some("completed".to_string()),
                },
                OutputItem::Reasoning {
                    id: "rsn_001".to_string(),
                    summary: vec![SummaryItem {
                        summary_type: "summary_text".to_string(),
                        text: "Thinking about the query.".to_string(),
                    }],
                },
                OutputItem::Message {
                    id: "msg_005".to_string(),
                    role: "assistant".to_string(),
                    status: Some("completed".to_string()),
                    content: vec![ContentItem::OutputText {
                        text: "Final answer.".to_string(),
                        annotations: vec![],
                    }],
                },
            ],
        );
        // WebSearchCall + Reasoning = 2; Message is excluded
        assert_eq!(response.research_step_count(), 2);
    }

    #[test]
    fn reasoning_summaries_extracted() {
        let response = make_response(
            ResponseStatus::Completed,
            vec![
                OutputItem::Reasoning {
                    id: "rsn_001".to_string(),
                    summary: vec![SummaryItem {
                        summary_type: "summary_text".to_string(),
                        text: "First reasoning step.".to_string(),
                    }],
                },
                OutputItem::Reasoning {
                    id: "rsn_002".to_string(),
                    summary: vec![
                        SummaryItem {
                            summary_type: "summary_text".to_string(),
                            text: "Second reasoning step A.".to_string(),
                        },
                        SummaryItem {
                            summary_type: "summary_text".to_string(),
                            text: "Second reasoning step B.".to_string(),
                        },
                    ],
                },
            ],
        );
        let summaries = response.reasoning_summaries();
        assert_eq!(summaries.len(), 3);
        assert_eq!(summaries[0], "First reasoning step.");
        assert_eq!(summaries[1], "Second reasoning step A.");
        assert_eq!(summaries[2], "Second reasoning step B.");
    }

    #[test]
    fn reasoning_summaries_skips_empty() {
        let response = make_response(
            ResponseStatus::Completed,
            vec![
                OutputItem::Reasoning {
                    id: "rsn_001".to_string(),
                    summary: vec![], // empty — should contribute nothing
                },
                OutputItem::Reasoning {
                    id: "rsn_002".to_string(),
                    summary: vec![SummaryItem {
                        summary_type: "summary_text".to_string(),
                        text: "Only this one.".to_string(),
                    }],
                },
            ],
        );
        let summaries = response.reasoning_summaries();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0], "Only this one.");
    }

    #[test]
    fn is_terminal_completed() {
        let r = make_response(ResponseStatus::Completed, vec![]);
        assert!(r.is_terminal());
    }

    #[test]
    fn is_terminal_failed() {
        let r = make_response(ResponseStatus::Failed, vec![]);
        assert!(r.is_terminal());
    }

    #[test]
    fn is_terminal_incomplete() {
        let r = make_response(ResponseStatus::Incomplete, vec![]);
        assert!(r.is_terminal());
    }

    #[test]
    fn is_terminal_queued() {
        let r = make_response(ResponseStatus::Queued, vec![]);
        assert!(!r.is_terminal());
    }

    #[test]
    fn is_terminal_in_progress() {
        let r = make_response(ResponseStatus::InProgress, vec![]);
        assert!(!r.is_terminal());
    }

    // -----------------------------------------------------------------------
    // OutputItem deserialization
    // -----------------------------------------------------------------------

    #[test]
    fn output_item_message_deser() {
        let json = json!({
            "type": "message",
            "id": "msg_abc",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "Hello world",
                    "annotations": []
                }
            ]
        });
        let item: OutputItem = from_value(json).expect("deserialize message OutputItem");
        match item {
            OutputItem::Message {
                id,
                role,
                status,
                content,
            } => {
                assert_eq!(id, "msg_abc");
                assert_eq!(role, "assistant");
                assert_eq!(status.as_deref(), Some("completed"));
                assert_eq!(content.len(), 1);
                match &content[0] {
                    ContentItem::OutputText { text, annotations } => {
                        assert_eq!(text, "Hello world");
                        assert!(annotations.is_empty());
                    }
                    _ => panic!("expected OutputText content"),
                }
            }
            _ => panic!("expected Message variant"),
        }
    }

    #[test]
    fn output_item_web_search_deser() {
        let json = json!({
            "type": "web_search_call",
            "id": "ws_xyz",
            "status": "completed"
        });
        let item: OutputItem = from_value(json).expect("deserialize web_search_call OutputItem");
        match item {
            OutputItem::WebSearchCall { id, status } => {
                assert_eq!(id, "ws_xyz");
                assert_eq!(status.as_deref(), Some("completed"));
            }
            _ => panic!("expected WebSearchCall variant"),
        }
    }

    #[test]
    fn output_item_reasoning_deser() {
        let json = json!({
            "type": "reasoning",
            "id": "rsn_001",
            "summary": [
                {
                    "type": "summary_text",
                    "text": "I am thinking about the problem."
                }
            ]
        });
        let item: OutputItem = from_value(json).expect("deserialize reasoning OutputItem");
        match item {
            OutputItem::Reasoning { id, summary } => {
                assert_eq!(id, "rsn_001");
                assert_eq!(summary.len(), 1);
                assert_eq!(summary[0].summary_type, "summary_text");
                assert_eq!(summary[0].text, "I am thinking about the problem.");
            }
            _ => panic!("expected Reasoning variant"),
        }
    }

    #[test]
    fn output_item_unknown_forward_compat() {
        // A type string the current code doesn't know about should not error
        let json = json!({
            "type": "some_future_item_type_v99",
            "id": "future_001",
            "data": "whatever"
        });
        let item: OutputItem = from_value(json).expect("deserialize unknown OutputItem");
        assert!(
            matches!(item, OutputItem::Unknown),
            "expected Unknown variant for unrecognized type"
        );
    }

    // -----------------------------------------------------------------------
    // Full response parsing
    // -----------------------------------------------------------------------

    #[test]
    fn full_response_json_deser() {
        let json_str = r#"
        {
            "id": "resp_abc123",
            "object": "response",
            "status": "completed",
            "model": "o3-deep-research-2025-06-26",
            "created_at": 1700000000.0,
            "output": [
                {
                    "type": "reasoning",
                    "id": "rsn_001",
                    "summary": [
                        {
                            "type": "summary_text",
                            "text": "Analyzing the research question."
                        }
                    ]
                },
                {
                    "type": "web_search_call",
                    "id": "ws_001",
                    "status": "completed"
                },
                {
                    "type": "web_search_call",
                    "id": "ws_002",
                    "status": "completed"
                },
                {
                    "type": "reasoning",
                    "id": "rsn_002",
                    "summary": [
                        {
                            "type": "summary_text",
                            "text": "Synthesizing findings from search results."
                        }
                    ]
                },
                {
                    "type": "message",
                    "id": "msg_001",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Based on my research, here are the findings.",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://example.com/article",
                                    "title": "Example Article",
                                    "start_index": 0,
                                    "end_index": 5
                                },
                                {
                                    "type": "url_citation",
                                    "url": "https://other.org/paper",
                                    "title": "Research Paper",
                                    "start_index": 10,
                                    "end_index": 20
                                }
                            ]
                        }
                    ]
                }
            ],
            "usage": {
                "input_tokens": 1500,
                "output_tokens": 800,
                "total_tokens": 2300
            }
        }
        "#;

        let response: Response = from_str(json_str).expect("deserialize full Response");

        // Top-level fields
        assert_eq!(response.id, "resp_abc123");
        assert_eq!(response.object, "response");
        assert_eq!(response.status, ResponseStatus::Completed);
        assert_eq!(response.model, "o3-deep-research-2025-06-26");
        assert_eq!(response.output.len(), 5);
        assert!(response.is_terminal());

        // Usage
        let usage = response.usage.as_ref().expect("usage should be present");
        assert_eq!(usage.input_tokens, 1500);
        assert_eq!(usage.output_tokens, 800);
        assert_eq!(usage.total_tokens, 2300);

        // final_report
        let report = response
            .final_report()
            .expect("final_report should be Some");
        assert_eq!(report, "Based on my research, here are the findings.");

        // citations
        let citations = response.citations();
        assert_eq!(citations.len(), 2);
        assert_eq!(
            citations[0].url.as_deref(),
            Some("https://example.com/article")
        );
        assert_eq!(citations[0].title.as_deref(), Some("Example Article"));
        assert_eq!(citations[1].url.as_deref(), Some("https://other.org/paper"));

        // research_step_count: 2 web searches + 2 reasoning = 4; message excluded
        assert_eq!(response.research_step_count(), 4);

        // reasoning_summaries
        let summaries = response.reasoning_summaries();
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0], "Analyzing the research question.");
        assert_eq!(summaries[1], "Synthesizing findings from search results.");
    }
}
