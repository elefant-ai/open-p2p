pub mod capture;

use anyhow::Context;
use iced::Task;
use std::sync::atomic::AtomicBool;
use tracing::info;

use crate::{
    paths::get_paths,
    sound::FileSource,
    widgets::{self, meta_data::capture_device_specs_from_target},
};

#[derive(Debug, Clone)]
#[allow(clippy::enum_variant_names)]
pub enum Message {
    /// Toggle recording
    ToggleRecording,
    ToggleRecordingWithInference,
    ToggleModelControl,
}

#[derive(Debug)]
pub struct State {
    pub running: bool,
    capture: capture::Capture,
}

impl State {
    pub fn new() -> (Self, Task<Message>) {
        (
            Self {
                capture: capture::Capture::new().unwrap(),
                running: false,
            },
            Task::none(),
        )
    }

    pub fn reset(&mut self) {
        self.capture.reset();
        self.running = false;
    }
}

fn handle_recording(top_state: &mut crate::App, with_inference: bool) -> Task<crate::Message> {
    let state = &mut top_state.handler;
    if !state.running {
        if top_state.target.is_none() {
            top_state.error = Some("No target selected".to_string());
            return Task::none();
        }
        if top_state.saved_state.enable_mic_audio && top_state.mic.is_none() {
            top_state.error = Some("No mic selected".to_string());
            return Task::none();
        }

        let target = top_state.target.as_ref().unwrap().window;
        let mic = if top_state.saved_state.enable_mic_audio {
            top_state.mic.clone()
        } else {
            None
        };

        let id = uuid::Uuid::now_v7();
        let path = get_paths().recordings_dir.join(id.to_string());

        if !path.exists() {
            std::fs::create_dir(&path).unwrap();
        }

        let (sec, subsec_nanos) = id.get_timestamp().unwrap().to_unix();
        let ms_time = (sec * 1000) + (subsec_nanos as u64 / 1_000_000);

        let mut meta_data = match widgets::meta_data::meta_data_from_state(&top_state.saved_state) {
            Ok(meta_data) => meta_data,
            Err(err) => {
                tracing::error!("Error getting meta data: {:?}", err);
                top_state.error = Some(format!("Error getting meta data: {err}"));
                FileSource::CaptureFailed.play();
                return Task::none();
            }
        };

        meta_data.id = id.to_string();
        meta_data.timestamp = ms_time as i64;
        meta_data.frames_per_second = 20.;
        meta_data.capture_device_specs = match capture_device_specs_from_target(&target) {
            Ok(specs) => Some(specs),
            Err(err) => {
                tracing::error!("Error getting capture device specs: {:?}", err);
                FileSource::CaptureFailed.play();
                top_state.error = Some(format!("Error getting capture device specs: {err}"));
                return Task::none();
            }
        };

        top_state
            .saved_state
            .recent_env
            .push(top_state.saved_state.env.trim().to_string());
        top_state.saved_state.recent_env.dedup();
        top_state
            .saved_state
            .recent_env_subtype
            .push(top_state.saved_state.env_subtype.trim().to_string());
        top_state.saved_state.recent_env_subtype.dedup();

        if let Err(err) = state.capture.start_capture(
            id,
            target,
            path,
            meta_data,
            with_inference,
            mic,
            top_state.saved_state.mic_volume,
        ) {
            FileSource::CaptureFailed.play();
            tracing::error!("Error starting capture: {:?}", err);
            top_state.error = Some(format!("Error starting capture: {err}"));
            state.running = false;
        } else {
            top_state.current_uuid = Some(id);
            crate::widgets::system_info::update(
                &mut top_state.system_info,
                widgets::system_info::SystemUpdate::SetId(Some(id)),
            );

            state.running = true;

            top_state.error = None;

            // enable to run the capture for 60 seconds and then stop it. used for checking file size
            #[cfg(feature = "encoding-check")]
            std::thread::spawn({
                move || {
                    std::thread::sleep(std::time::Duration::from_secs(60));
                    send_message(crate::Message::HotKey(crate::HotKey::ToggleRecording));
                }
            });
        }
    } else {
        state.running = false;
        crate::widgets::system_info::update(
            &mut top_state.system_info,
            widgets::system_info::SystemUpdate::SetId(None),
        );

        if let Err(e) = state.capture.stop_capture() {
            top_state.error = Some(format!("Error stopping capture: {e}"));
        }
        if let Some(id) = top_state.current_uuid {
            let new_snap = top_state.metrics_handle.snapshot();
            top_state.snapshot.merge(new_snap);
            let saved = crate::performance::recording::RecordingStorage::get_data_from_snapshot(
                &top_state.snapshot,
                id,
            );

            return Task::future(async move {
                info!("Saving recording {} data metadata", id);
                if let Err(e) = saved.save(id).await {
                    tracing::error!("Error saving recording data: {:?}", e);
                }
            })
            .discard();
        }
    }
    Task::none()
}

pub fn update(top_state: &mut crate::App, message: Message) -> Task<crate::Message> {
    match message {
        Message::ToggleRecording => handle_recording(top_state, false),
        Message::ToggleRecordingWithInference => handle_recording(top_state, true),
        Message::ToggleModelControl => {
            top_state.handler.capture.toggle_model_control();
            Task::none()
        }
    }
}

/// Enable mouse acceleration on exit
pub static ENABLE_ON_EXIT: AtomicBool = AtomicBool::new(false);

#[must_use]
pub struct OnDrop;

impl Drop for OnDrop {
    fn drop(&mut self) {
        if ENABLE_ON_EXIT.load(std::sync::atomic::Ordering::Relaxed) {
            set_mouse_acceleration(true).unwrap();
        }
    }
}

pub fn init_mouse() -> Result<OnDrop, anyhow::Error> {
    let enabled = capture::get_mouse_acceleration()?;
    if enabled {
        ENABLE_ON_EXIT.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Err(e) = capture::set_mouse_acceleration(false) {
            eprintln!("Failed to disable mouse acceleration: {e:?}");
            return Err(anyhow::anyhow!(
                "Failed to disable mouse acceleration: {e:?}"
            ));
        }
    }

    Ok(OnDrop)
}

fn set_mouse_acceleration(enabled: bool) -> Result<(), anyhow::Error> {
    capture::set_mouse_acceleration(enabled)
        .context(format!("Failed to set mouse acceleration to {enabled}"))
}
