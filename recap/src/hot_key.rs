use crate::{
    App, handler,
    input_manager::{Event, listen},
    upload,
};
use iced::{
    Subscription, Task,
    futures::{SinkExt, Stream, StreamExt as _},
    stream,
};
use input_codes::Keycode;
use tracing::error;

#[derive(Debug, Clone)]
#[allow(clippy::enum_variant_names)]
pub enum HotKey {
    ToggleRecording,
    ToggleRecordingWithInference,
    TogglePlayback,
    ToggleModelControl,
}

pub fn update(app: &mut App, hotkey: HotKey) -> Task<crate::Message> {
    match hotkey {
        HotKey::ToggleRecording => {
            return handler::update(app, handler::Message::ToggleRecording);
        }
        HotKey::ToggleRecordingWithInference => {
            return handler::update(app, handler::Message::ToggleRecordingWithInference);
        }
        HotKey::ToggleModelControl => {
            return handler::update(app, handler::Message::ToggleModelControl);
        }
        HotKey::TogglePlayback => {
            let first_id = app.uploader.files.first().map(|x| x.0);
            if let Some(id) = first_id {
                return upload::update(
                    app,
                    upload::Message::RunBack(crate::paths::get_annotation_path(&id)),
                )
                .handle(app, crate::Message::Uploader);
            } else {
                app.error = Some("No files to play".to_string());
            }
        }
    }
    Task::none()
}

pub fn subscription(_model: &crate::App) -> Subscription<crate::Message> {
    Subscription::run(watch_hotkeys)
}

/// Watch for hotkey events and send corresponding messages only when all keys in a hotkey combination are pressed and released.
fn watch_hotkeys() -> impl Stream<Item = crate::Message> {
    stream::channel(
        10,
        |mut output: iced::futures::channel::mpsc::Sender<crate::Message>| async move {
            let (mut tx, mut rx) = iced::futures::channel::mpsc::channel(10);

            let id = listen(move |event, _| {
                if matches!(event.event, Event::KeyboardInput { .. }) && !event.simulated {
                    if let Err(err) = tx.try_send(event.clone()) {
                        error!("Error sending hotkey event: {:?}", err);
                    }
                }
            });

            let mut currently_pressed = std::collections::HashSet::new();
            let mut pending_message = None;

            while let Some(event) = rx.next().await {
                match event.event {
                    Event::KeyboardInput { pressed: true, key } => {
                        currently_pressed.insert(key);

                        // Check if any hotkey combination is fully pressed
                        if TOGGLE_RECORDING_HOTKEY
                            .iter()
                            .all(|key| currently_pressed.contains(key))
                        {
                            pending_message = Some(crate::Message::HotKey(HotKey::ToggleRecording));
                        }
                        #[cfg(feature = "inference")]
                        if TOGGLE_RECORDING_WITH_INFERENCE_HOTKEY
                            .iter()
                            .all(|key| currently_pressed.contains(key))
                        {
                            pending_message =
                                Some(crate::Message::HotKey(HotKey::ToggleRecordingWithInference));
                        }
                        #[cfg(feature = "playback")]
                        if TOGGLE_PLAYBACK_HOTKEY
                            .iter()
                            .all(|key| currently_pressed.contains(key))
                        {
                            pending_message = Some(crate::Message::HotKey(HotKey::TogglePlayback));
                        }
                    }
                    Event::KeyboardInput {
                        pressed: false,
                        key,
                    } => {
                        currently_pressed.remove(&key);

                        // If we have a pending message and all keys for any hotkey combination are released,
                        // send the message
                        if let Some(message) = pending_message.take() {
                            let should_send = match message {
                                crate::Message::HotKey(HotKey::ToggleRecording) => {
                                    !TOGGLE_RECORDING_HOTKEY
                                        .iter()
                                        .any(|k| currently_pressed.contains(k))
                                }
                                #[cfg(feature = "inference")]
                                crate::Message::HotKey(HotKey::ToggleRecordingWithInference) => {
                                    !TOGGLE_RECORDING_WITH_INFERENCE_HOTKEY
                                        .iter()
                                        .any(|k| currently_pressed.contains(k))
                                }
                                #[cfg(feature = "playback")]
                                crate::Message::HotKey(HotKey::TogglePlayback) => {
                                    !TOGGLE_PLAYBACK_HOTKEY
                                        .iter()
                                        .any(|k| currently_pressed.contains(k))
                                }
                                _ => false,
                            };

                            if should_send {
                                if let Err(err) = output.send(message).await {
                                    error!("Error sending message: {:?}", err);
                                }
                            } else {
                                // Put the message back if not all keys are released
                                pending_message = Some(message);
                            }
                        }
                    }
                    _ => {}
                }
            }

            crate::input_manager::remove_listener(id);
        },
    )
}

/// The hotkey to toggle recording
pub const TOGGLE_RECORDING_HOTKEY: &[Keycode] = &[Keycode::RightBracket];

/// the hotkey to toggle recording with inference
pub const TOGGLE_RECORDING_WITH_INFERENCE_HOTKEY: &[Keycode] =
    &[Keycode::RightBracket, Keycode::LeftShift];

/// The hotkey to toggle model control
pub const TOGGLE_MODEL_CONTROL_HOTKEY: &[Keycode] = &[Keycode::LeftBracket];

/// the hotkey to toggle playback
pub const TOGGLE_PLAYBACK_HOTKEY: &[Keycode] = &[Keycode::BackSlash];
