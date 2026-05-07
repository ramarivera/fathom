//! CLI interface for fathom.
//!
//! Provides subcommands for standalone usage:
//! - `research` — start a deep research job and optionally wait for results
//! - `status`   — check the status of an existing job
//! - `results`  — retrieve full results of a completed job
//! - `serve`    — run the MCP server (stdio or HTTP)

use clap::{Parser, Subcommand, ValueEnum};

/// Fathom — Deep Research from the command line.
///
/// A CLI + MCP server wrapping the OpenAI Deep Research API.
/// Preserves full intermediate research output (web searches,
/// reasoning, code runs) that other tools discard.
#[derive(Debug, Parser)]
#[command(name = "fathom", version, about, long_about = None)]
pub struct Cli {
    /// OpenAI API key. Can also be set via OPENAI_API_KEY env var.
    #[arg(long, env = "OPENAI_API_KEY", global = true, hide_env_values = true)]
    pub api_key: Option<String>,

    /// OpenAI API base URL (for proxies or testing).
    #[arg(long, env = "OPENAI_API_BASE", global = true)]
    pub api_base: Option<String>,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Start a new deep research job.
    Research {
        /// The research query or question.
        query: String,

        /// Model to use.
        #[arg(long, short, default_value = "o3")]
        model: ModelChoice,

        /// ISO country code for web search localization (e.g., "US", "GB").
        #[arg(long)]
        country: Option<String>,

        /// City for web search localization.
        #[arg(long)]
        city: Option<String>,

        /// Region/state for web search localization.
        #[arg(long)]
        region: Option<String>,

        /// Web search context depth. Deep Research currently supports only "medium".
        #[arg(long)]
        search_context_size: Option<ContextSize>,

        /// Enable code interpreter during research.
        #[arg(long)]
        code_interpreter: bool,

        /// Developer instructions for the model.
        #[arg(long)]
        instructions: Option<String>,

        /// Wait for the job to complete and print results.
        #[arg(long, short)]
        wait: bool,
    },

    /// Check the status of a research job.
    Status {
        /// Response ID (starts with "resp_").
        response_id: String,
    },

    /// Retrieve full results of a research job.
    Results {
        /// Response ID (starts with "resp_").
        response_id: String,

        /// Exclude intermediate steps (show only the final report).
        #[arg(long)]
        no_steps: bool,
    },

    /// Run as an MCP server.
    Serve {
        /// Transport mode.
        #[arg(long, short, default_value = "stdio")]
        transport: TransportMode,

        /// Bind address for HTTP transport.
        #[arg(long, default_value = "127.0.0.1:3100")]
        bind: String,

        /// Bearer token for HTTP transport authentication.
        /// Can also be set via FATHOM_AUTH_TOKEN env var.
        #[arg(long, env = "FATHOM_AUTH_TOKEN", hide_env_values = true)]
        auth_token: Option<String>,
    },
}

/// Deep research model choices.
#[derive(Debug, Clone, ValueEnum)]
pub enum ModelChoice {
    /// o3-deep-research (full power, slower)
    O3,
    /// o4-mini-deep-research (faster, cheaper)
    O4Mini,
}

/// Web search context size choices supported by Deep Research.
#[derive(Debug, Clone, ValueEnum)]
pub enum ContextSize {
    Medium,
}

/// MCP server transport mode.
#[derive(Debug, Clone, ValueEnum)]
pub enum TransportMode {
    /// JSON-RPC over stdin/stdout (for local agents).
    Stdio,
    /// Streamable HTTP (for remote access).
    Http,
}
