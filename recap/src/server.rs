use crate::snap_shot_state::StateSnapshot;
use anyhow::Error as AnyhowError;
use glam::DVec2;
use http_body_util::{BodyExt, Full};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode, body::Incoming as IncomingBody};
use hyper_util::rt::TokioIo;
use iced::futures::channel::mpsc;
use iced::futures::{SinkExt, StreamExt};
use iced::{Subscription, stream};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use tokio::net::TcpListener;
use tracing::{error, info, warn};
#[cfg(target_os = "windows")]
use win_programs::WinProgram;
use window_handling::WindowInfo;

use crate::Message;

/// Configuration for the server
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub port: u16,
    pub bind_address: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            bind_address: "127.0.0.1".to_string(),
        }
    }
}

/// Messages that can be sent to the server from external clients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServerMessage {
    /// Refresh the device list
    Refresh,
    /// List available target windows
    ListTargets,
    /// Set the target window by title
    SetTarget { title: String },
    /// Set the task name
    SetTask { task: String },
    /// Set the environment
    SetEnv { env: String },
    /// Set environment subtype
    SetEnvSubtype { env_subtype: String },
    /// Set the user
    SetUser { user: String },
    /// Save current settings
    SaveSettings,
    /// Toggle recording
    ToggleRecording,
    /// Toggle recording with inference
    ToggleRecordingWithInference,
    /// Toggle playback
    TogglePlayback,
    /// Exit the application
    Exit,
    /// Get current status
    GetStatus,
    /// Set window size
    SetWindowSize { width: i32, height: i32 },
    /// Get current window size
    GetWindowSize,
    /// Set window position
    SetWindowPosition { x: i32, y: i32 },
    /// Get current window position
    GetWindowPosition,
    /// Move mouse to absolute position
    MoveMouse { x: f64, y: f64 },
    /// Playback annotations
    Playback { path: String },
    /// Toggle model control
    ToggleModelControl,
    /// Start Program
    StartProgram { name: String, args: Vec<String> },
}

/// Response from the server to the client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServerResponse {
    /// Command executed successfully
    Success { message: String },
    /// Command failed with error
    Error { error: String },
    /// Current status information
    Status {
        recording: bool,
        current_target: Option<String>,
        available_targets: Vec<String>,
    },
    /// Recording toggle response with path information
    RecordingToggled {
        recording: bool,
        path: Option<String>,
        message: String,
    },
}

/// Create a subscription for the server when feature is enabled
pub fn subscription() -> Subscription<Message> {
    Subscription::run(|| {
        stream::channel(100, |output: mpsc::Sender<Message>| async move {
            let config = ServerConfig::default();
            if let Err(e) = start_server(config, output.clone()).await {
                error!("Server failed to start: {}", e);
            }
        })
    })
}

/// Start the server and handle incoming connections
async fn start_server(
    config: ServerConfig,
    message_sender: mpsc::Sender<Message>,
) -> Result<(), AnyhowError> {
    let addr = format!("{}:{}", config.bind_address, config.port);
    let listener = TcpListener::bind(&addr).await?;

    info!("HTTP server listening on http://{}", addr);

    loop {
        match listener.accept().await {
            Ok((stream, addr)) => {
                let sender_clone = message_sender.clone();
                let io = TokioIo::new(stream);

                tokio::spawn(async move {
                    if let Err(e) = http1::Builder::new()
                        .serve_connection(
                            io,
                            service_fn(move |req| handle_request(req, sender_clone.clone())),
                        )
                        .await
                    {
                        error!("Error serving connection from {}: {}", addr, e);
                    }
                });
            }
            Err(e) => {
                error!("Failed to accept connection: {}", e);
            }
        }
    }
}

/// Query the current application state
async fn query_state(
    message_sender: &mut mpsc::Sender<Message>,
) -> Result<StateSnapshot, AnyhowError> {
    let (tx, mut rx) = mpsc::channel(1); // Use buffer size of 1 since we only expect one response
    message_sender.send(Message::QueryState(tx)).await?;

    // Use a timeout to avoid hanging indefinitely
    let state = tokio::time::timeout(std::time::Duration::from_secs(5), rx.next())
        .await
        .map_err(|_| anyhow::anyhow!("Timeout waiting for state response"))?
        .ok_or_else(|| anyhow::anyhow!("Channel closed"))?;

    Ok(state)
}

/// Handle HTTP requests
async fn handle_request(
    req: Request<IncomingBody>,
    message_sender: mpsc::Sender<Message>,
) -> Result<Response<Full<hyper::body::Bytes>>, Infallible> {
    match (req.method(), req.uri().path()) {
        (&hyper::Method::POST, "/command") => match req.collect().await {
            Ok(body) => {
                let body_bytes = body.to_bytes();
                match std::str::from_utf8(&body_bytes) {
                    Ok(body_str) => {
                        let mut sender_clone = message_sender.clone();
                        let response = process_message(body_str, &mut sender_clone).await;
                        let response_json = match serde_json::to_string(&response) {
                            Ok(json) => json,
                            Err(e) => {
                                let error_response = ServerResponse::Error {
                                    error: format!("Failed to serialize response: {e}"),
                                };
                                serde_json::to_string(&error_response).unwrap_or_else(|_| {
                                        r#"{"Error":{"error":"Failed to serialize error response"}}"#.to_string()
                                    })
                            }
                        };

                        Ok(Response::builder()
                            .status(StatusCode::OK)
                            .header("Content-Type", "application/json")
                            .header("Access-Control-Allow-Origin", "*")
                            .header("Access-Control-Allow-Methods", "POST, OPTIONS")
                            .header("Access-Control-Allow-Headers", "Content-Type")
                            .body(Full::new(response_json.into()))
                            .unwrap())
                    }
                    Err(e) => {
                        let error_response = ServerResponse::Error {
                            error: format!("Invalid UTF-8 in request body: {e}"),
                        };
                        let response_json = serde_json::to_string(&error_response).unwrap();

                        Ok(Response::builder()
                            .status(StatusCode::BAD_REQUEST)
                            .header("Content-Type", "application/json")
                            .header("Access-Control-Allow-Origin", "*")
                            .body(Full::new(response_json.into()))
                            .unwrap())
                    }
                }
            }
            Err(e) => {
                let error_response = ServerResponse::Error {
                    error: format!("Failed to read request body: {e}"),
                };
                let response_json = serde_json::to_string(&error_response).unwrap();

                Ok(Response::builder()
                    .status(StatusCode::BAD_REQUEST)
                    .header("Content-Type", "application/json")
                    .header("Access-Control-Allow-Origin", "*")
                    .body(Full::new(response_json.into()))
                    .unwrap())
            }
        },
        (&hyper::Method::OPTIONS, "/command") => {
            // Handle CORS preflight requests
            Ok(Response::builder()
                .status(StatusCode::OK)
                .header("Access-Control-Allow-Origin", "*")
                .header("Access-Control-Allow-Methods", "POST, OPTIONS")
                .header("Access-Control-Allow-Headers", "Content-Type")
                .body(Full::new("".into()))
                .unwrap())
        }
        _ => {
            let error_response = ServerResponse::Error {
                error: "Not found. Use POST /command".to_string(),
            };
            let response_json = serde_json::to_string(&error_response).unwrap();

            Ok(Response::builder()
                .status(StatusCode::NOT_FOUND)
                .header("Content-Type", "application/json")
                .header("Access-Control-Allow-Origin", "*")
                .body(Full::new(response_json.into()))
                .unwrap())
        }
    }
}

/// Process a message from a client and return a response
async fn process_message(
    message: &str,
    message_sender: &mut mpsc::Sender<Message>,
) -> ServerResponse {
    match serde_json::from_str::<ServerMessage>(message) {
        Ok(server_message) => match handle_server_message(server_message, message_sender).await {
            Ok(response) => response,
            Err(e) => ServerResponse::Error {
                error: format!("Failed to handle message: {e}"),
            },
        },
        Err(e) => {
            warn!("Failed to parse message '{}': {}", message, e);
            ServerResponse::Error {
                error: format!("Invalid JSON message: {e}"),
            }
        }
    }
}

/// Handle a parsed server message and convert it to application messages
async fn handle_server_message(
    message: ServerMessage,
    message_sender: &mut mpsc::Sender<Message>,
) -> Result<ServerResponse, AnyhowError> {
    match message {
        ServerMessage::Refresh => {
            if let Err(e) = message_sender.send(Message::Refresh).await {
                return Ok(ServerResponse::Error {
                    error: format!("Failed to send refresh message: {e}"),
                });
            }
            Ok(ServerResponse::Success {
                message: "Device list refreshed".to_string(),
            })
        }
        ServerMessage::ListTargets => match query_state(message_sender).await {
            Ok(state) => {
                let available_targets: Vec<String> =
                    state.devices.iter().map(|d| d.title.clone()).collect();
                let current_target = state.target.as_ref().map(|t| t.title.clone());

                if available_targets.is_empty() {
                    Ok(ServerResponse::Success {
                        message: "No targets available. Try refreshing the device list."
                            .to_string(),
                    })
                } else {
                    let target_list = available_targets.join(", ");
                    let message = match current_target {
                        Some(current) => {
                            format!("Available targets: [{target_list}]. Current: {current}")
                        }
                        None => format!(
                            "Available targets: [{target_list}]. No target currently selected."
                        ),
                    };
                    Ok(ServerResponse::Success { message })
                }
            }
            Err(e) => Ok(ServerResponse::Error {
                error: format!("Failed to query application state: {e}"),
            }),
        },
        ServerMessage::SetTarget { title } => {
            // First, get the current state to find the target by title
            match query_state(message_sender).await {
                Ok(state) => {
                    // Look for a device with the matching title
                    if let Some(target) = state.devices.iter().find(|device| device.title == title)
                    {
                        if let Err(e) = message_sender
                            .send(Message::SetTarget(target.clone()))
                            .await
                        {
                            return Ok(ServerResponse::Error {
                                error: format!("Failed to send set target message: {e}"),
                            });
                        }
                        Ok(ServerResponse::Success {
                            message: format!("Target set to: {title}"),
                        })
                    } else {
                        // Target not found, list available targets
                        let available_titles: Vec<String> =
                            state.devices.iter().map(|d| d.title.clone()).collect();
                        Ok(ServerResponse::Error {
                            error: format!(
                                "Target '{}' not found. Available targets: [{}]",
                                title,
                                available_titles.join(", ")
                            ),
                        })
                    }
                }
                Err(e) => Ok(ServerResponse::Error {
                    error: format!("Failed to query application state: {e}"),
                }),
            }
        }
        ServerMessage::SetTask { task } => {
            if let Err(e) = message_sender.send(Message::SetTask(task.clone())).await {
                return Ok(ServerResponse::Error {
                    error: format!("Failed to send set task message: {e}"),
                });
            }
            Ok(ServerResponse::Success {
                message: format!("Task set to: {task}"),
            })
        }
        ServerMessage::SetEnv { env } => {
            if let Err(e) = message_sender.send(Message::SetEnv(env.clone())).await {
                return Ok(ServerResponse::Error {
                    error: format!("Failed to send set env message: {e}"),
                });
            }
            Ok(ServerResponse::Success {
                message: format!("Environment set to: {env}"),
            })
        }
        ServerMessage::SetEnvSubtype { env_subtype } => {
            if let Err(e) = message_sender
                .send(Message::SetEnvSubtype(env_subtype.clone()))
                .await
            {
                return Ok(ServerResponse::Error {
                    error: format!("Failed to send set env subtype message: {e}"),
                });
            }
            Ok(ServerResponse::Success {
                message: format!("Environment subtype set to: {env_subtype}"),
            })
        }
        ServerMessage::SetUser { user } => {
            if let Err(e) = message_sender.send(Message::SetUser(user.clone())).await {
                return Ok(ServerResponse::Error {
                    error: format!("Failed to send set user message: {e}"),
                });
            }
            Ok(ServerResponse::Success {
                message: format!("User set to: {user}"),
            })
        }
        ServerMessage::SaveSettings => {
            if let Err(e) = message_sender.send(Message::SaveSettings).await {
                return Ok(ServerResponse::Error {
                    error: format!("Failed to send save settings message: {e}"),
                });
            }
            Ok(ServerResponse::Success {
                message: "Settings saved".to_string(),
            })
        }
        ServerMessage::ToggleRecording => {
            if let Err(e) = message_sender
                .send(Message::HotKey(crate::hot_key::HotKey::ToggleRecording))
                .await
            {
                return Ok(ServerResponse::Error {
                    error: format!("Failed to send toggle recording message: {e}"),
                });
            }

            // Query state after toggle to get recording status and path
            match query_state(message_sender).await {
                Ok(state) => {
                    let path = if state.recording && state.current_uuid.is_some() {
                        let uuid = state.current_uuid.unwrap();
                        Some(
                            crate::paths::get_paths()
                                .recordings_dir
                                .join(uuid.to_string())
                                .to_string_lossy()
                                .to_string(),
                        )
                    } else {
                        None
                    };

                    Ok(ServerResponse::RecordingToggled {
                        recording: state.recording,
                        path,
                        message: if state.recording {
                            "Recording started".to_string()
                        } else {
                            "Recording stopped".to_string()
                        },
                    })
                }
                Err(e) => Ok(ServerResponse::Error {
                    error: format!("Failed to query state after toggle: {e}"),
                }),
            }
        }
        ServerMessage::ToggleRecordingWithInference => {
            if let Err(e) = message_sender
                .send(Message::HotKey(
                    crate::hot_key::HotKey::ToggleRecordingWithInference,
                ))
                .await
            {
                return Ok(ServerResponse::Error {
                    error: format!("Failed to send toggle recording with inference message: {e}"),
                });
            }

            // Query state after toggle to get recording status and path
            match query_state(message_sender).await {
                Ok(state) => {
                    let path = if state.recording && state.current_uuid.is_some() {
                        let uuid = state.current_uuid.unwrap();
                        Some(
                            crate::paths::get_paths()
                                .recordings_dir
                                .join(uuid.to_string())
                                .to_string_lossy()
                                .to_string(),
                        )
                    } else {
                        None
                    };

                    Ok(ServerResponse::RecordingToggled {
                        recording: state.recording,
                        path,
                        message: if state.recording {
                            "Recording with inference started".to_string()
                        } else {
                            "Recording with inference stopped".to_string()
                        },
                    })
                }
                Err(e) => Ok(ServerResponse::Error {
                    error: format!("Failed to query state after toggle: {e}"),
                }),
            }
        }
        ServerMessage::TogglePlayback => {
            if let Err(e) = message_sender
                .send(Message::HotKey(crate::hot_key::HotKey::TogglePlayback))
                .await
            {
                return Ok(ServerResponse::Error {
                    error: format!("Failed to send toggle playback message: {e}"),
                });
            }
            Ok(ServerResponse::Success {
                message: "Playback toggled".to_string(),
            })
        }
        ServerMessage::Exit => {
            if let Err(e) = message_sender.send(Message::Exit).await {
                return Ok(ServerResponse::Error {
                    error: format!("Failed to send exit message: {e}"),
                });
            }
            Ok(ServerResponse::Success {
                message: "Application will exit".to_string(),
            })
        }
        ServerMessage::GetStatus => match query_state(message_sender).await {
            Ok(state) => {
                let current_target = state.target.as_ref().map(|t| t.title.clone());
                let available_targets: Vec<String> =
                    state.devices.iter().map(|d| d.title.clone()).collect();

                Ok(ServerResponse::Status {
                    recording: state.recording,
                    current_target,
                    available_targets,
                })
            }
            Err(e) => Ok(ServerResponse::Error {
                error: format!("Failed to query application state: {e}"),
            }),
        },
        ServerMessage::SetWindowSize { width, height } => {
            // First, get the current state to ensure we have a target
            match query_state(message_sender).await {
                Ok(state) => {
                    if state.target.is_some() {
                        // Use the WindowSize message to apply the size
                        if let Err(e) = message_sender
                            .send(Message::WindowSize(
                                crate::widgets::window_size::WindowSizeMessage::SetPresetSize(
                                    width, height,
                                ),
                            ))
                            .await
                        {
                            return Ok(ServerResponse::Error {
                                error: format!("Failed to send set window size message: {e}"),
                            });
                        }
                        if let Err(e) = message_sender
                            .send(Message::WindowSize(
                                crate::widgets::window_size::WindowSizeMessage::ApplySize,
                            ))
                            .await
                        {
                            return Ok(ServerResponse::Error {
                                error: format!("Failed to apply window size: {e}"),
                            });
                        }
                        Ok(ServerResponse::Success {
                            message: format!("Window size set to {width}x{height}"),
                        })
                    } else {
                        Ok(ServerResponse::Error {
                            error: "No target window selected. Please set a target first."
                                .to_string(),
                        })
                    }
                }
                Err(e) => Ok(ServerResponse::Error {
                    error: format!("Failed to query application state: {e}"),
                }),
            }
        }
        ServerMessage::GetWindowSize => match query_state(message_sender).await {
            Ok(state) => {
                if let Some(target) = &state.target {
                    match target.window.size() {
                        Ok((width, height)) => Ok(ServerResponse::Success {
                            message: format!("Current window size: {width}x{height}"),
                        }),
                        Err(_) => Ok(ServerResponse::Error {
                            error: "Unable to get window size".to_string(),
                        }),
                    }
                } else {
                    Ok(ServerResponse::Error {
                        error: "No target window selected".to_string(),
                    })
                }
            }
            Err(e) => Ok(ServerResponse::Error {
                error: format!("Failed to query application state: {e}"),
            }),
        },
        ServerMessage::SetWindowPosition { x, y } => {
            // First, get the current state to ensure we have a target
            match query_state(message_sender).await {
                Ok(state) => {
                    if state.target.is_some() {
                        // Use the WindowSize message to apply the position
                        if let Err(e) = message_sender
                            .send(Message::WindowSize(
                                crate::widgets::window_size::WindowSizeMessage::SetPresetPosition(
                                    x, y,
                                ),
                            ))
                            .await
                        {
                            return Ok(ServerResponse::Error {
                                error: format!("Failed to send set window position message: {e}"),
                            });
                        }
                        if let Err(e) = message_sender
                            .send(Message::WindowSize(
                                crate::widgets::window_size::WindowSizeMessage::ApplyPosition,
                            ))
                            .await
                        {
                            return Ok(ServerResponse::Error {
                                error: format!("Failed to apply window position: {e}"),
                            });
                        }
                        Ok(ServerResponse::Success {
                            message: format!("Window position set to ({x}, {y})"),
                        })
                    } else {
                        Ok(ServerResponse::Error {
                            error: "No target window selected. Please set a target first."
                                .to_string(),
                        })
                    }
                }
                Err(e) => Ok(ServerResponse::Error {
                    error: format!("Failed to query application state: {e}"),
                }),
            }
        }
        ServerMessage::GetWindowPosition => match query_state(message_sender).await {
            Ok(state) => {
                if let Some(target) = &state.target {
                    match target.window.position() {
                        Ok((x, y)) => Ok(ServerResponse::Success {
                            message: format!("Current window position: ({x}, {y})"),
                        }),
                        Err(_) => Ok(ServerResponse::Error {
                            error: "Unable to get window position".to_string(),
                        }),
                    }
                } else {
                    Ok(ServerResponse::Error {
                        error: "No target window selected".to_string(),
                    })
                }
            }
            Err(e) => Ok(ServerResponse::Error {
                error: format!("Failed to query application state: {e}"),
            }),
        },
        ServerMessage::MoveMouse { x, y } => {
            // Import the simulate module to use mouse movement
            crate::input_manager::simulate::simulate_mouse_absolute(DVec2::new(x, y));
            Ok(ServerResponse::Success {
                message: format!("Mouse moved to position ({x}, {y})"),
            })
        }
        ServerMessage::Playback { path } => {
            let path_buf = std::path::PathBuf::from(path.clone());
            if let Err(e) = message_sender.send(Message::RunBack(path_buf)).await {
                return Ok(ServerResponse::Error {
                    error: format!("Failed to send playback message: {e}"),
                });
            }
            Ok(ServerResponse::Success {
                message: format!("Playback started for path: {path}"),
            })
        }
        ServerMessage::ToggleModelControl => {
            crate::external::send_message(crate::Message::HotKey(
                crate::hot_key::HotKey::ToggleModelControl,
            ));
            Ok(ServerResponse::Success {
                message: "Model control toggled".to_string(),
            })
        }
        #[cfg(target_os = "windows")]
        ServerMessage::StartProgram { name, args } => match WinProgram::new(name.clone()) {
            Ok(program) => {
                let arg_refs: Vec<&str> = args.iter().map(std::string::String::as_str).collect();
                if let Err(e) = program.start_with_args(&arg_refs) {
                    return Ok(ServerResponse::Error {
                        error: format!("Failed to start program '{name}': {e}"),
                    });
                }
                Ok(ServerResponse::Success {
                    message: format!("Program '{name}' started successfully"),
                })
            }
            Err(e) => Ok(ServerResponse::Error {
                error: format!("Error launching program '{name}': {e}"),
            }),
        },
        #[cfg(not(target_os = "windows"))]
        ServerMessage::StartProgram { name, args } => Ok(ServerResponse::Error {
            error: "Starting programs is only supported on Windows".to_string(),
        }),
    }
}
