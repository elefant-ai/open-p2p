use std::{io::Write as _, path::Path};

use anyhow::Context as _;
use glam::IVec2;
use rayon::iter::{IntoParallelIterator, ParallelIterator as _};
use video_annotation_proto::video_annotation::{
    GamePadAction, GamePadAxisEvent, GamePadButtonEvent, GamePadButtons, GamePadTriggerEvent,
    InputEvent, KeyboardEvent, MouseButtonEvent, Stick, VideoAnnotationMetadata, input_event,
};

use crate::input_manager::{DeviceEvent, Event, collect_input_frames, game_pad};

#[derive(Debug, Clone)]
pub struct InputFrame {
    pub time: std::time::SystemTime,
    pub user_keys: Vec<input_codes::Keycode>,
    pub system_keys: Vec<input_codes::Keycode>,
    pub inference_running: bool,
    pub user_mouse: InputFrameMouse,
    pub system_mouse: InputFrameMouse,
    pub game_pad: Option<game_pad::GamePad>,
    pub timeline: Vec<DeviceEvent>,
}

#[derive(Debug, Clone)]
pub struct InputFrameMouse {
    pub delta: IVec2,
    pub mouse_pos: IVec2,
    pub buttons: Vec<input_codes::Button>,
    pub scroll: IVec2,
}

impl InputFrame {
    pub fn get_codes() -> Self {
        collect_input_frames()
    }
}

pub fn save_input_state(
    input_state: Vec<InputFrame>,
    path: &Path,
    meta_data: VideoAnnotationMetadata,
    start_time: std::time::SystemTime,
    voice_events: Vec<(bool, std::time::SystemTime)>,
) -> Result<(), anyhow::Error> {
    use video_annotation_proto::prost::Message;
    use video_annotation_proto::video_annotation::{
        FrameAnnotation, KeyboardAction, LowLevelAction, VideoAnnotation,
    };

    let frame_annotations = input_state.into_par_iter().map(|frame| {
        let to_action =
            |keys: Vec<input_codes::Keycode>,
             mouse: InputFrameMouse,
             is_known: bool,
             game_pad: Option<game_pad::GamePad>| LowLevelAction {
                mouse: Some(video_annotation_proto::video_annotation::MouseAction {
                    mouse_absolute_px: Some(mouse.mouse_pos.into()),
                    // mouse relative position in fraction of window size TODO setup
                    // mouse_relative: Some((mouse_pos * window_size).into()),
                    mouse_relative: None,
                    mouse_delta_px: Some(mouse.delta.into()),
                    scroll_delta_px: Some(mouse.scroll.into()),
                    buttons_down: mouse
                        .buttons
                        .iter()
                        .map(|b| match b {
                            input_codes::Button::Left => 0,
                            input_codes::Button::Right => 1,
                            input_codes::Button::Middle => 2,
                            input_codes::Button::Unknown(key) => *key as u32,
                        })
                        .map(|b| b.to_string())
                        .collect(),
                }),
                keyboard: if !keys.is_empty() {
                    Some(KeyboardAction {
                        keys: keys.iter().map(std::string::ToString::to_string).collect(),
                    })
                } else {
                    None
                },
                is_known,
                mouse_deprecated: None,
                game_pad: game_pad.map(|game_pad| GamePadAction {
                    buttons: Some(GamePadButtons {
                        south: game_pad.buttons.south,
                        north: game_pad.buttons.north,
                        east: game_pad.buttons.east,
                        west: game_pad.buttons.west,
                        dpad_up: game_pad.buttons.dpad_up,
                        dpad_down: game_pad.buttons.dpad_down,
                        dpad_left: game_pad.buttons.dpad_left,
                        dpad_right: game_pad.buttons.dpad_right,
                        start: game_pad.buttons.start,
                        select: game_pad.buttons.select,
                        left_bumper: game_pad.buttons.left_bumper,
                        right_bumper: game_pad.buttons.right_bumper,
                    }),
                    left_stick: Some(Stick {
                        x: game_pad.left_stick.x,
                        y: game_pad.left_stick.y,
                        pressed: game_pad.left_stick.pressed,
                    }),
                    right_stick: Some(Stick {
                        x: game_pad.right_stick.x,
                        y: game_pad.right_stick.y,
                        pressed: game_pad.right_stick.pressed,
                    }),
                    left_trigger: game_pad.triggers.left_trigger,
                    right_trigger: game_pad.triggers.right_trigger,
                }),
            };

        Ok(FrameAnnotation {
            user_action: Some(to_action(
                frame.user_keys,
                frame.user_mouse,
                !frame.inference_running,
                frame.game_pad,
            )),
            system_action: Some(to_action(
                frame.system_keys,
                frame.system_mouse,
                frame.inference_running,
                None,
            )),
            action_task: None,
            env_state: None,
            frame_text_annotation: vec![],
            frame_time: frame
                .time
                .duration_since(start_time)
                .context("Failed to calculate frame time")?
                .as_nanos() as u64,
            input_events: frame
                .timeline
                .into_iter()
                .filter_map(|event| {
                    let time = event.time;
                    let simulated = event.simulated;
                    let event = match event.event {
                        Event::MouseButton { pressed, button } => {
                            input_event::Event::MouseEvent(MouseButtonEvent {
                                button: button.to_string(),
                                pressed,
                            })
                        }
                        Event::KeyboardInput { pressed, key } => {
                            input_event::Event::KeyboardEvent(KeyboardEvent {
                                key: key.to_string(),
                                pressed,
                            })
                        }
                        Event::MouseMove(delta) => input_event::Event::MouseMoveEvent(delta.into()),
                        Event::MouseWheel(delta) => input_event::Event::WheelEvent(delta.into()),
                        Event::MouseDelta(delta) => {
                            input_event::Event::MouseDeltaEvent(delta.into())
                        }
                        Event::GamePadAction(event_type) => map_gamepad_event(event_type)?,
                    };
                    Some((time, event, simulated))
                })
                .map(|(time, event, simulated)| {
                    Ok(InputEvent {
                        event: Some(event),
                        simulated,
                        time: time
                            .duration_since(start_time)
                            .context("Failed to calculate input event time")?
                            .as_nanos() as u64,
                    })
                })
                .collect::<Result<_, anyhow::Error>>()?,
        })
    });

    let voice_events = voice_events
        .into_iter()
        .map(|(speaking, time)| {
            Ok(video_annotation_proto::video_annotation::VoiceEvent {
                speaking,
                time: time
                    .duration_since(start_time)
                    .context("Failed to calculate voice event time")?
                    .as_nanos() as u64,
            })
        })
        .collect::<Result<_, anyhow::Error>>()?;

    let annotation = VideoAnnotation {
        metadata: Some(meta_data),
        // Version configured in Cargo.toml [package.metadata.versions] section
        version: env!("PROTO_VERSION").parse().unwrap(),
        frame_annotations: frame_annotations.collect::<Result<Vec<_>, anyhow::Error>>()?,
        voice_events,
        ..VideoAnnotation::default()
    };

    let mut buf = Vec::with_capacity(annotation.encoded_len());
    annotation
        .encode(&mut buf)
        .context("Failed to encode input state")?;

    let mut file = std::fs::File::create(path.join("annotation.proto"))
        .context("Failed to create input state file")?;
    file.write_all(&buf)
        .context("Failed to write input state")?;
    Ok(())
}

fn map_gamepad_event(event: gilrs::EventType) -> Option<input_event::Event> {
    match event {
        gilrs::EventType::ButtonPressed(button, _) => Some(input_event::Event::GamePadButtonEvent(
            map_gamepad_buttons(button, true)?,
        )),
        gilrs::EventType::ButtonReleased(button, _) => Some(
            input_event::Event::GamePadButtonEvent(map_gamepad_buttons(button, false)?),
        ),
        gilrs::EventType::AxisChanged(axis, value, _) => {
            Some(input_event::Event::GamePadAxisEvent(GamePadAxisEvent {
                axis: match axis {
                    gilrs::Axis::LeftStickX => "left_stick_x".to_string(),
                    gilrs::Axis::LeftStickY => "left_stick_y".to_string(),
                    gilrs::Axis::RightStickX => "right_stick_x".to_string(),
                    gilrs::Axis::RightStickY => "right_stick_y".to_string(),
                    _ => return None,
                },
                value,
            }))
        }
        gilrs::EventType::ButtonChanged(button, value, _) => Some(
            input_event::Event::GamePadTriggerEvent(GamePadTriggerEvent {
                trigger: match button {
                    gilrs::Button::LeftTrigger2 => "left_trigger".to_string(),
                    gilrs::Button::RightTrigger2 => "right_trigger".to_string(),
                    _ => return None,
                },
                value,
            }),
        ),
        _ => None,
    }
}

fn map_gamepad_buttons(button: gilrs::Button, pressed: bool) -> Option<GamePadButtonEvent> {
    let button_name = match button {
        gilrs::Button::South => "south",
        gilrs::Button::East => "east",
        gilrs::Button::North => "north",
        gilrs::Button::West => "west",
        gilrs::Button::LeftTrigger => "left_trigger",
        gilrs::Button::RightTrigger => "right_trigger",
        gilrs::Button::Select => "select",
        gilrs::Button::Start => "start",
        gilrs::Button::LeftThumb => "left_stick",
        gilrs::Button::RightThumb => "right_stick",
        gilrs::Button::DPadUp => "dpad_up",
        gilrs::Button::DPadDown => "dpad_down",
        gilrs::Button::DPadLeft => "dpad_left",
        gilrs::Button::DPadRight => "dpad_right",
        _ => return None,
    };
    Some(GamePadButtonEvent {
        button: button_name.to_string(),
        pressed,
    })
}
