use iced::widget;
use video_annotation_proto::video_annotation::{VideoAnnotationEnv, VideoAnnotationMetadata};
use window_handling::{MonitorInfo as _, WindowInfo};

use crate::{Message, SavedState, input_manager::keyboard::keyboard_layout};

// Version now configured in Cargo.toml [package.metadata.versions] section
pub const RECAP_VERSION: &str = env!("RECAP_VERSION");
pub const GIT_COMMIT: &str = env!("GIT_COMMIT");

pub(crate) fn meta_data_from_state(
    saved_state: &SavedState,
) -> Result<VideoAnnotationMetadata, anyhow::Error> {
    // task is not used currently
    // if state.saved_state.task.is_empty() {
    //     return Err(anyhow::anyhow!("Task is empty"));
    // }
    if saved_state.user.trim().is_empty() {
        return Err(anyhow::anyhow!("User is empty"));
    }

    if saved_state.env.trim().is_empty() {
        return Err(anyhow::anyhow!("Env is empty"));
    }

    let meta_data = VideoAnnotationMetadata {
        env: Some(VideoAnnotationEnv {
            env: saved_state.env.trim().to_string(),
            env_subtype: saved_state.env_subtype.trim().to_string(),
            env_version: "".to_string(),
        }),
        user: saved_state.user.trim().to_string(),
        tasks: vec![saved_state.task.trim().to_string()],
        ..Default::default()
    };
    Ok(meta_data)
}

pub fn capture_device_specs_from_target(
    target: &dyn WindowInfo,
) -> Result<video_annotation_proto::video_annotation::CaptureDeviceSpecs, anyhow::Error> {
    let (width, height) = target.size()?;
    let monitor = target.current_monitor()?;
    let monitor_id = monitor.id() as i64;
    let monitor_size = monitor.size()?;
    let capture_device_specs = video_annotation_proto::video_annotation::CaptureDeviceSpecs {
        recap_version: RECAP_VERSION.parse().unwrap(),
        git_commit: GIT_COMMIT.to_string(),
        os: std::env::consts::OS.to_string(),
        keyboard_layout: keyboard_layout()?.to_string(),
        window_specs: Some(video_annotation_proto::video_annotation::WindowSpecs {
            width,
            height,
            scale: target.scale_factor(),
            dpi: target.dpi()? as i32,
            title: target.title().unwrap_or_default(),
            screen_id: monitor_id,
        }),
        screen_specs: vec![video_annotation_proto::video_annotation::ScreenSpecs {
            width: monitor_size.0,
            height: monitor_size.1,
            scale: monitor.scale_factor(),
            dpi: monitor.dpi()? as i32,
            id: monitor_id,
        }],
    };
    Ok(capture_device_specs)
}

pub(crate) fn set_meta_data(state: &crate::App) -> iced::Element<'_, Message> {
    widget::column![
        widget::row![
            widget::text_input("you name", &state.saved_state.user)
                .on_input(Message::SetUser)
                .width(150.0),
            widget::text_input("task", &state.saved_state.task)
                .on_input(Message::SetTask)
                .width(150.0),
        ]
        .spacing(10),
        widget::row![
            widget::text_input("env", &state.saved_state.env)
                .on_input(Message::SetEnv)
                .width(150.0),
            widget::text_input("env subtype", &state.saved_state.env_subtype)
                .on_input(Message::SetEnvSubtype)
                .width(150.0),
        ]
        .spacing(10),
        widget::button("Save settings")
            .on_press(Message::SaveSettings)
            .padding(10)
            .style(iced::widget::button::primary),
    ]
    .spacing(10)
    .into()
}
