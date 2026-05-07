//! Fathom entry point.
//!
//! Dispatches between CLI subcommands and MCP server modes.

use anyhow::{Context, Result};
use clap::Parser;
use rmcp::{transport::stdio, ServiceExt};
use tracing_subscriber::EnvFilter;

use fathom_mcp::api::client::{CreateOptions, DeepResearchClient};
use fathom_mcp::api::types::{DeepResearchModel, SearchContextSize};
use fathom_mcp::cli::{Cli, Command, ContextSize, ModelChoice, TransportMode};
use fathom_mcp::mcp::server::FathomServer;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing — stderr to avoid interfering with stdio MCP transport.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .with_ansi(true)
        .init();

    let cli = Cli::parse();

    let api_key = cli
        .api_key
        .as_deref()
        .or(std::env::var("OPENAI_API_KEY").ok().as_deref())
        .context("OPENAI_API_KEY is required (set via --api-key or env var)")?
        .to_string();

    let client = if let Some(base) = &cli.api_base {
        DeepResearchClient::with_base_url(&api_key, base)?
    } else {
        DeepResearchClient::new(&api_key)?
    };

    match cli.command {
        Command::Research {
            query,
            model,
            country,
            city,
            region,
            search_context_size,
            code_interpreter,
            instructions,
            wait,
        } => {
            cmd_research(
                &client,
                &query,
                model,
                country,
                city,
                region,
                search_context_size,
                code_interpreter,
                instructions,
                wait,
            )
            .await
        }
        Command::Status { response_id } => cmd_status(&client, &response_id).await,
        Command::Results {
            response_id,
            no_steps,
        } => cmd_results(&client, &response_id, no_steps).await,
        Command::Serve {
            transport,
            bind,
            auth_token,
        } => cmd_serve(client, transport, &bind, auth_token).await,
    }
}

// ---------------------------------------------------------------------------
// CLI subcommands
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn cmd_research(
    client: &DeepResearchClient,
    query: &str,
    model: ModelChoice,
    country: Option<String>,
    city: Option<String>,
    region: Option<String>,
    search_context_size: Option<ContextSize>,
    code_interpreter: bool,
    instructions: Option<String>,
    wait: bool,
) -> Result<()> {
    let model = match model {
        ModelChoice::O3 => DeepResearchModel::O3,
        ModelChoice::O4Mini => DeepResearchModel::O4Mini,
    };

    let context_size = search_context_size.map(|s| match s {
        ContextSize::Medium => SearchContextSize::Medium,
    });

    let options = CreateOptions {
        country,
        city,
        region,
        search_context_size: context_size,
        code_interpreter,
        instructions,
        store: Some(true),
        metadata: None,
    };

    eprintln!("Creating deep research job with model {}...", model);
    let response = client.create(query, model, options).await?;
    eprintln!("Job created: {}", response.id);
    eprintln!("Status: {}", response.status);

    if wait {
        eprintln!("Waiting for completion (this typically takes 5-30 minutes)...");
        let response = client.poll_until_complete(&response.id).await?;

        eprintln!("Status: {}", response.status);
        eprintln!("Research steps: {}", response.research_step_count());

        if let Some(report) = response.final_report() {
            println!("{report}");
        } else {
            eprintln!("No report available.");
        }

        let citations = response.citations();
        if !citations.is_empty() {
            eprintln!("\n--- Citations ({}) ---", citations.len());
            for ann in &citations {
                if let Some(url) = &ann.url {
                    let title = ann.title.as_deref().unwrap_or("(untitled)");
                    eprintln!("  - {title}: {url}");
                }
            }
        }

        if let Some(usage) = &response.usage {
            eprintln!(
                "\nTokens: {} input, {} output, {} total",
                usage.input_tokens, usage.output_tokens, usage.total_tokens
            );
        }
    } else {
        // Print the response ID so the user can poll manually.
        println!("{}", response.id);
        eprintln!("Use `fathom status {}` to check progress.", response.id);
    }

    Ok(())
}

async fn cmd_status(client: &DeepResearchClient, response_id: &str) -> Result<()> {
    let response = client.retrieve(response_id).await?;

    eprintln!("Response: {}", response.id);
    eprintln!("Status: {}", response.status);
    eprintln!("Research steps: {}", response.research_step_count());

    let summaries = response.reasoning_summaries();
    if !summaries.is_empty() {
        eprintln!("\nReasoning summaries:");
        for (i, summary) in summaries.iter().enumerate() {
            eprintln!("  {}. {}", i + 1, summary);
        }
    }

    if let Some(error) = &response.error {
        eprintln!("\nError: {} — {}", error.code, error.message);
    }

    if response.is_terminal() {
        eprintln!(
            "\nJob complete. Use `fathom results {}` to get the report.",
            response_id
        );
    } else {
        eprintln!("\nJob still running. Check again in 10-30 seconds.");
    }

    Ok(())
}

async fn cmd_results(client: &DeepResearchClient, response_id: &str, no_steps: bool) -> Result<()> {
    let response = client.retrieve(response_id).await?;

    if !response.is_terminal() {
        eprintln!(
            "Warning: job is still {} — results may be incomplete.",
            response.status
        );
    }

    // Print the report to stdout.
    if let Some(report) = response.final_report() {
        println!("{report}");
    } else {
        eprintln!("No report available yet.");
    }

    // Print metadata to stderr.
    let citations = response.citations();
    if !citations.is_empty() {
        eprintln!("\n--- Citations ({}) ---", citations.len());
        for ann in &citations {
            if let Some(url) = &ann.url {
                let title = ann.title.as_deref().unwrap_or("(untitled)");
                eprintln!("  - {title}: {url}");
            }
        }
    }

    if !no_steps {
        eprintln!("\nResearch steps: {}", response.research_step_count());

        let summaries = response.reasoning_summaries();
        if !summaries.is_empty() {
            eprintln!("\nReasoning summaries:");
            for (i, summary) in summaries.iter().enumerate() {
                eprintln!("  {}. {}", i + 1, summary);
            }
        }

        // Print the full output as JSON to stderr for inspection.
        eprintln!("\n--- Full output (JSON) ---");
        let output_json = serde_json::to_string_pretty(&response.output).unwrap_or_default();
        eprintln!("{output_json}");
    }

    if let Some(usage) = &response.usage {
        eprintln!(
            "\nTokens: {} input, {} output, {} total",
            usage.input_tokens, usage.output_tokens, usage.total_tokens
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// MCP server
// ---------------------------------------------------------------------------

async fn cmd_serve(
    client: DeepResearchClient,
    transport: TransportMode,
    bind: &str,
    auth_token: Option<String>,
) -> Result<()> {
    match transport {
        TransportMode::Stdio => {
            tracing::info!("Starting fathom MCP server (stdio transport)");

            let server = FathomServer::new(client);
            let service = server
                .serve(stdio())
                .await
                .context("failed to start stdio MCP server")?;

            service.waiting().await?;
            Ok(())
        }
        TransportMode::Http => {
            use rmcp::transport::streamable_http_server::{
                session::local::LocalSessionManager, StreamableHttpServerConfig,
                StreamableHttpService,
            };

            let bind_addr = bind.to_string();
            tracing::info!(%bind_addr, "Starting fathom MCP server (HTTP transport)");

            let ct = tokio_util::sync::CancellationToken::new();

            let service = StreamableHttpService::new(
                move || Ok(FathomServer::new(client.clone())),
                LocalSessionManager::default().into(),
                StreamableHttpServerConfig {
                    cancellation_token: ct.child_token(),
                    ..Default::default()
                },
            );

            let router = axum::Router::new().nest_service("/mcp", service);

            // Add bearer token auth middleware if configured.
            let router = if let Some(token) = auth_token {
                use axum::http::StatusCode;
                use axum::middleware::{self, Next};
                use axum::response::IntoResponse;

                let router = router.layer(middleware::from_fn(
                    move |req: axum::http::Request<axum::body::Body>, next: Next| {
                        let expected = token.clone();
                        async move {
                            let auth_header = req
                                .headers()
                                .get(axum::http::header::AUTHORIZATION)
                                .and_then(|v| v.to_str().ok())
                                .map(|s| s.to_string());

                            match auth_header {
                                Some(header) if header == format!("Bearer {expected}") => {
                                    next.run(req).await
                                }
                                _ => StatusCode::UNAUTHORIZED.into_response(),
                            }
                        }
                    },
                ));
                tracing::info!("Bearer token authentication enabled");
                router
            } else {
                router
            };

            let tcp_listener = tokio::net::TcpListener::bind(&bind_addr)
                .await
                .context("failed to bind TCP listener")?;

            tracing::info!("Listening on http://{bind_addr}/mcp");

            axum::serve(tcp_listener, router)
                .with_graceful_shutdown(async move {
                    tokio::signal::ctrl_c().await.ok();
                    tracing::info!("Shutting down...");
                    ct.cancel();
                })
                .await
                .context("HTTP server error")?;

            Ok(())
        }
    }
}
