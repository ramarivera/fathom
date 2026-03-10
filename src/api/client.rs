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
        let mut tools: Vec<Tool> = Vec::new();

        // Web search is always included for deep research.
        let user_location = if options.country.is_some()
            || options.city.is_some()
            || options.region.is_some()
        {
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
