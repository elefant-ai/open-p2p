use std::{path::Path, str::FromStr as _, time::Duration};

use anyhow::Context as _;
use input_codes::{Button, Keycode};
use video_annotation_proto::video_annotation::{GamePadAction, VideoAnnotation};
use video_inference_grpc::prost::Message as _;

use crate::input_manager::{
    lift_simulated_keys,
    simulate::{simulate_key, simulate_mouse_button, simulate_mouse_delta, simulate_mouse_scroll},
};

pub async fn play_back_annotations(file: impl AsRef<Path>) -> Result<(), anyhow::Error> {
    let proto_data = tokio::fs::read(file)
        .await
        .context("Failed to annotation file")?;

    let proto = VideoAnnotation::decode(proto_data.as_slice())?;

    let meta_data = proto.metadata.as_ref().context("No metadata found")?;
    let fps = meta_data.frames_per_second;
    let frame_gap = Duration::from_secs_f32(1.0 / fps);
    let mut interval = tokio::time::interval(frame_gap);

    let mut game_pad_playback = GamePadPlayBack::new();

    let mut keys_pressed: Vec<String> = Vec::new();
    let mut mouse_buttons_pressed: Vec<String> = Vec::new();

    for (idx, frame) in proto.frame_annotations.into_iter().enumerate() {
        interval.tick().await;

        if let Some(action) = frame.user_action {
            let keys = action.keyboard.map(|k| k.keys).unwrap_or_default();
            process_keys(keys, &mut keys_pressed);
            let mouse_buttons = action
                .mouse
                .as_ref()
                .map(|m| m.buttons_down.clone())
                .unwrap_or_default();
            process_mouse_buttons(mouse_buttons, &mut mouse_buttons_pressed);
            if let Some(mouse) = action.mouse {
                if let Some(pos) = mouse.mouse_delta_px {
                    simulate_mouse_delta(pos.into());
                }
                if let Some(scroll) = mouse.scroll_delta_px {
                    simulate_mouse_scroll(scroll.into());
                }
            }
            if let Some(game_pad) = action.game_pad {
                game_pad_playback.playback(game_pad);
            }
        } else {
            tracing::warn!("No user actions found in frame {}", idx);
        }
    }

    lift_simulated_keys();

    Ok(())
}

struct GamePadPlayBack {
    target: vigem_client::Xbox360Wired<vigem_client::Client>,
    game_pad: vigem_client::XGamepad,
}

impl GamePadPlayBack {
    pub fn new() -> Self {
        let client = vigem_client::Client::connect().unwrap();

        let mut target =
            vigem_client::Xbox360Wired::new(client, vigem_client::TargetId::XBOX360_WIRED);

        target.plugin().unwrap();

        target.wait_ready().unwrap();

        Self {
            target,
            game_pad: vigem_client::XGamepad::default(),
        }
    }

    fn normalize_trigger(&self, trigger: f32) -> u8 {
        let normalized = trigger * u8::MAX as f32;
        normalized as u8
    }

    fn normalize_stick(&self, stick: f32) -> i16 {
        let normalized = stick * i16::MAX as f32;
        normalized as i16
    }

    fn playback(&mut self, inputs: GamePadAction) {
        self.game_pad.left_trigger = self.normalize_trigger(inputs.left_trigger);
        self.game_pad.right_trigger = self.normalize_trigger(inputs.right_trigger);
        let left_stick = inputs.left_stick.unwrap();
        let right_stick = inputs.right_stick.unwrap();
        self.game_pad.thumb_lx = self.normalize_stick(left_stick.x);
        self.game_pad.thumb_ly = self.normalize_stick(left_stick.y);
        if left_stick.pressed {
            self.game_pad.buttons.raw |= vigem_client::XButtons::LTHUMB;
        } else {
            self.game_pad.buttons.raw &= !vigem_client::XButtons::LTHUMB;
        }
        self.game_pad.thumb_rx = self.normalize_stick(right_stick.x);
        self.game_pad.thumb_ry = self.normalize_stick(right_stick.y);
        if right_stick.pressed {
            self.game_pad.buttons.raw |= vigem_client::XButtons::RTHUMB;
        } else {
            self.game_pad.buttons.raw &= !vigem_client::XButtons::RTHUMB;
        }
        let buttons = inputs.buttons.unwrap();
        if buttons.south {
            self.game_pad.buttons.raw |= vigem_client::XButtons::A;
        } else {
            self.game_pad.buttons.raw &= !vigem_client::XButtons::A;
        }
        if buttons.north {
            self.game_pad.buttons.raw |= vigem_client::XButtons::Y;
        } else {
            self.game_pad.buttons.raw &= !vigem_client::XButtons::Y;
        }
        if buttons.east {
            self.game_pad.buttons.raw |= vigem_client::XButtons::B;
        } else {
            self.game_pad.buttons.raw &= !vigem_client::XButtons::B;
        }
        if buttons.west {
            self.game_pad.buttons.raw |= vigem_client::XButtons::X;
        } else {
            self.game_pad.buttons.raw &= !vigem_client::XButtons::X;
        }
        if buttons.dpad_up {
            self.game_pad.buttons.raw |= vigem_client::XButtons::UP;
        } else {
            self.game_pad.buttons.raw &= !vigem_client::XButtons::UP;
        }
        if buttons.dpad_down {
            self.game_pad.buttons.raw |= vigem_client::XButtons::DOWN;
        } else {
            self.game_pad.buttons.raw &= !vigem_client::XButtons::DOWN;
        }
        if buttons.dpad_left {
            self.game_pad.buttons.raw |= vigem_client::XButtons::LEFT;
        } else {
            self.game_pad.buttons.raw &= !vigem_client::XButtons::LEFT;
        }
        if buttons.dpad_right {
            self.game_pad.buttons.raw |= vigem_client::XButtons::RIGHT;
        } else {
            self.game_pad.buttons.raw &= !vigem_client::XButtons::RIGHT;
        }
        if buttons.start {
            self.game_pad.buttons.raw |= vigem_client::XButtons::START;
        } else {
            self.game_pad.buttons.raw &= !vigem_client::XButtons::START;
        }
        if buttons.select {
            self.game_pad.buttons.raw |= vigem_client::XButtons::BACK;
        } else {
            self.game_pad.buttons.raw &= !vigem_client::XButtons::BACK;
        }
        if buttons.left_bumper {
            self.game_pad.buttons.raw |= vigem_client::XButtons::LB;
        } else {
            self.game_pad.buttons.raw &= !vigem_client::XButtons::LB;
        }
        if buttons.right_bumper {
            self.game_pad.buttons.raw |= vigem_client::XButtons::RB;
        } else {
            self.game_pad.buttons.raw &= !vigem_client::XButtons::RB;
        }
        let _ = self.target.update(&self.game_pad);
    }
}

fn process_mouse_buttons(mouse_buttons: Vec<String>, previous_mouse_buttons: &mut Vec<String>) {
    // if the mouse button is not in the previous buttons, simulate a button press
    mouse_buttons
        .iter()
        .filter(|button| !previous_mouse_buttons.contains(button))
        .for_each(|button| {
            if let Ok(button) = Button::from_str(button) {
                simulate_mouse_button(button, true);
            }
        });

    // if the mouse button is not in the current buttons, simulate a button release
    previous_mouse_buttons
        .iter()
        .filter(|button| !mouse_buttons.contains(button))
        .for_each(|button| {
            if let Ok(button) = Button::from_str(button) {
                simulate_mouse_button(button, false);
            }
        });

    // update the previous mouse buttons to the current buttons
    *previous_mouse_buttons = mouse_buttons;
}

fn process_keys(keys: Vec<String>, previous_keys: &mut Vec<String>) {
    // if the key is not in the previous keys, simulate a key press
    keys.iter()
        .filter(|key| !previous_keys.contains(key))
        .for_each(|key| {
            if let Ok(key) = Keycode::from_str(key) {
                simulate_key(key, true);
            }
        });

    // if the key is not in the current keys, simulate a key release
    previous_keys
        .iter()
        .filter(|key| !keys.contains(key))
        .for_each(|key| {
            if let Ok(key) = Keycode::from_str(key) {
                simulate_key(key, false);
            }
        });

    // update the previous keys to the current keys
    *previous_keys = keys;
}
