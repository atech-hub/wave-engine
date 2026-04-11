//! HTTP server — Axum router, shared state, auth, graceful shutdown.

use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post};
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Json, Response};
use tokio::net::TcpListener;

use crate::common::wave_model::WavePacketModel;
use crate::common::dims::Dims;
use super::api_types::{ErrorResponse, ErrorDetail};
use super::handlers;
use super::prompt::Vocab;

/// Shared application state.
pub struct AppState {
    pub model: Arc<WavePacketModel>,
    pub vocab: Arc<Vocab>,
    pub dims: Dims,
    pub stencil: Arc<crate::fft_ode::StencilFft>,
    pub model_name: String,
    pub api_key: Option<String>,
    pub host: String,
    pub port: u16,
    pub memory: Option<std::sync::Mutex<kerr_memory::memory::WaveMemory>>,
    pub memory_path: Option<String>,
}

/// Bearer token auth middleware.
async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Response {
    let Some(ref expected_key) = state.api_key else {
        return next.run(req).await;
    };

    let auth_header = req.headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(header) if header.starts_with("Bearer ") => {
            let token = &header[7..];
            if token == expected_key {
                return next.run(req).await;
            }
        }
        _ => {}
    }

    let err = ErrorResponse {
        error: ErrorDetail {
            message: "Invalid or missing API key".to_string(),
            r#type: "authentication_error".to_string(),
        },
    };
    (StatusCode::UNAUTHORIZED, Json(err)).into_response()
}

/// Start the HTTP server.
pub fn run_server(state: Arc<AppState>) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(run(state));
}

async fn run(state: Arc<AppState>) {
    let host = state.host.clone();
    let port = state.port;
    let has_auth = state.api_key.is_some();

    let protected = Router::new()
        .route("/v1/chat/completions", post(handlers::handle_chat_completion))
        .route("/v1/models", get(handlers::handle_models))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

    let app = Router::new()
        .merge(protected)
        .route("/health", get(handlers::handle_health))
        .with_state(state);

    let addr = format!("{host}:{port}");
    let listener = match TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("ERROR: cannot bind to {addr} — {e}");
            std::process::exit(1);
        }
    };

    println!("  Server listening on http://{addr}");
    println!("  Auth: {}", if has_auth { "API key required" } else { "NONE" });
    println!("  Endpoints:");
    println!("    POST /v1/chat/completions");
    println!("    GET  /v1/models");
    println!("    GET  /health");
    println!("  Press Ctrl+C to stop");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();

    println!("  Server stopped.");
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c().await.expect("failed to install Ctrl+C handler");
    println!("\n  Shutting down...");
}
