//! Fathom — Deep Research API client and MCP server.
//!
//! This crate provides:
//! - `api::client::DeepResearchClient` — HTTP client for the OpenAI Responses API
//! - `api::types` — comprehensive type definitions with full intermediate output parsing
//! - `mcp::server::FathomServer` — MCP server exposing research tools
//! - `cli` — CLI interface definitions

pub mod api;
pub mod cli;
pub mod mcp;
