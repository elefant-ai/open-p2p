use parking_lot::Mutex;
use std::sync::{Arc, LazyLock};

use crate::input_manager::{DeviceEvent, Event, send_device_event};

static GILRS: LazyLock<Mutex<GamePadState>> = LazyLock::new(|| Mutex::new(GamePadState::new()));

struct GamePadState {
    game_pad: Arc<Mutex<Option<GamePad>>>,
}

impl GamePadState {
    pub fn new() -> Self {
        let mut gilrs = gilrs::Gilrs::new().expect("Failed to initialize Gilrs");
        let gamepad_state: Arc<Mutex<Option<GamePad>>> = Arc::new(Mutex::new(None));

        std::thread::spawn({
            let gamepad_state = gamepad_state.clone();
            move || {
                let mut game_pad_id: Option<gilrs::GamepadId> = None;
                while let Some(event) = gilrs.next_event_blocking(None) {
                    send_device_event(DeviceEvent {
                        time: event.time,
                        event: Event::GamePadAction(event.event),
                        simulated: false,
                    });
                    if let gilrs::EventType::Connected = event.event {
                        game_pad_id = Some(event.id);
                    } else if let gilrs::EventType::Disconnected = event.event
                        && let Some(id) = game_pad_id
                        && event.id == id
                    {
                        let maybeid = gilrs.gamepads().next().map(|(id, _)| id);
                        if let Some(new_id) = maybeid {
                            game_pad_id = Some(new_id);
                            // have to reinitialize the gamepad state
                            let new_gamepad = GamePad::from_game_pad(gilrs.gamepad(new_id));
                            *gamepad_state.lock() = Some(new_gamepad);
                        } else {
                            game_pad_id = None;
                            *gamepad_state.lock() = None;
                        }
                    } else if game_pad_id.is_none() {
                        // if we don't have a gamepad id yet, just take the first event's id
                        game_pad_id = Some(event.id);
                    }

                    let mut gamepad_state = gamepad_state.lock();

                    if let Some(id) = game_pad_id {
                        let gamepad_state = if let Some(gamepad) = gamepad_state.as_mut() {
                            gamepad
                        } else {
                            let new_gamepad = GamePad::from_game_pad(gilrs.gamepad(id));
                            *gamepad_state = Some(new_gamepad);
                            gamepad_state.as_mut().unwrap()
                        };

                        match event.event {
                            gilrs::EventType::ButtonPressed(button, _) => match button {
                                gilrs::Button::South => gamepad_state.buttons.south = true,
                                gilrs::Button::North => gamepad_state.buttons.north = true,
                                gilrs::Button::East => gamepad_state.buttons.east = true,
                                gilrs::Button::West => gamepad_state.buttons.west = true,
                                gilrs::Button::DPadUp => gamepad_state.buttons.dpad_up = true,
                                gilrs::Button::DPadDown => gamepad_state.buttons.dpad_down = true,
                                gilrs::Button::DPadLeft => gamepad_state.buttons.dpad_left = true,
                                gilrs::Button::DPadRight => gamepad_state.buttons.dpad_right = true,
                                gilrs::Button::Start => gamepad_state.buttons.start = true,
                                gilrs::Button::Select => gamepad_state.buttons.select = true,
                                gilrs::Button::LeftTrigger => {
                                    gamepad_state.buttons.left_bumper = true;
                                }
                                gilrs::Button::RightTrigger => {
                                    gamepad_state.buttons.right_bumper = true;
                                }
                                gilrs::Button::Mode => {} // Mode button is not handled
                                gilrs::Button::LeftThumb => gamepad_state.left_stick.pressed = true,
                                gilrs::Button::RightThumb => {
                                    gamepad_state.right_stick.pressed = true;
                                }
                                _ => {}
                            },
                            gilrs::EventType::ButtonReleased(button, _) => {
                                match button {
                                    gilrs::Button::South => gamepad_state.buttons.south = false,
                                    gilrs::Button::North => gamepad_state.buttons.north = false,
                                    gilrs::Button::East => gamepad_state.buttons.east = false,
                                    gilrs::Button::West => gamepad_state.buttons.west = false,
                                    gilrs::Button::DPadUp => gamepad_state.buttons.dpad_up = false,
                                    gilrs::Button::DPadDown => {
                                        gamepad_state.buttons.dpad_down = false;
                                    }
                                    gilrs::Button::DPadLeft => {
                                        gamepad_state.buttons.dpad_left = false;
                                    }
                                    gilrs::Button::DPadRight => {
                                        gamepad_state.buttons.dpad_right = false;
                                    }
                                    gilrs::Button::Start => gamepad_state.buttons.start = false,
                                    gilrs::Button::Select => gamepad_state.buttons.select = false,
                                    gilrs::Button::LeftTrigger => {
                                        gamepad_state.buttons.left_bumper = false;
                                    }
                                    gilrs::Button::RightTrigger => {
                                        gamepad_state.buttons.right_bumper = false;
                                    }
                                    gilrs::Button::Mode => {} // Mode button is not handled
                                    gilrs::Button::LeftThumb => {
                                        gamepad_state.left_stick.pressed = false;
                                    }
                                    gilrs::Button::RightThumb => {
                                        gamepad_state.right_stick.pressed = false;
                                    }
                                    _ => {}
                                }
                            }
                            gilrs::EventType::ButtonChanged(button, value, _) => match button {
                                gilrs::Button::LeftTrigger2 => {
                                    gamepad_state.triggers.left_trigger = value;
                                }
                                gilrs::Button::RightTrigger2 => {
                                    gamepad_state.triggers.right_trigger = value;
                                }
                                _ => {}
                            },
                            gilrs::EventType::AxisChanged(axis, value, _) => match axis {
                                gilrs::Axis::LeftStickX => gamepad_state.left_stick.x = value,
                                gilrs::Axis::LeftStickY => gamepad_state.left_stick.y = value,
                                gilrs::Axis::RightStickX => gamepad_state.right_stick.x = value,
                                gilrs::Axis::RightStickY => gamepad_state.right_stick.y = value,
                                _ => {}
                            },
                            _ => {}
                        }
                    } else {
                        *gamepad_state = None;
                    }
                }
            }
        });

        Self {
            game_pad: gamepad_state,
        }
    }

    pub fn get_game_pad_state(&mut self) -> Option<GamePad> {
        let game_pad = self.game_pad.lock();
        *game_pad
    }
}

/// Get the current game pad state if it exists
pub fn get_state() -> Option<GamePad> {
    let mut input = GILRS.lock();
    input.get_game_pad_state()
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct GamePad {
    pub buttons: Buttons,
    pub triggers: Triggers,
    pub left_stick: LeftStick,
    pub right_stick: RightStick,
}

impl GamePad {
    pub fn from_game_pad(input: gilrs::Gamepad<'_>) -> Self {
        Self {
            buttons: Buttons::from_game_pad(input),
            triggers: Triggers::from_game_pad(input),
            left_stick: LeftStick::from_game_pad(input),
            right_stick: RightStick::from_game_pad(input),
        }
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct LeftStick {
    pub x: f32,
    pub y: f32,
    pub pressed: bool,
}

impl LeftStick {
    pub fn from_game_pad(input: gilrs::Gamepad<'_>) -> Self {
        Self {
            x: input.value(gilrs::Axis::LeftStickX),
            y: input.value(gilrs::Axis::LeftStickY),
            pressed: input.is_pressed(gilrs::Button::LeftThumb),
        }
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct RightStick {
    pub x: f32,
    pub y: f32,
    pub pressed: bool,
}

impl RightStick {
    pub fn from_game_pad(input: gilrs::Gamepad<'_>) -> Self {
        Self {
            x: input.value(gilrs::Axis::RightStickX),
            y: input.value(gilrs::Axis::RightStickY),
            pressed: input.is_pressed(gilrs::Button::RightThumb),
        }
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Triggers {
    pub left_trigger: f32,
    pub right_trigger: f32,
}

impl Triggers {
    pub fn from_game_pad(input: gilrs::Gamepad<'_>) -> Self {
        Self {
            left_trigger: input
                .button_data(gilrs::Button::LeftTrigger2)
                .map(gilrs::ev::state::ButtonData::value)
                .unwrap_or(0.0),
            right_trigger: input
                .button_data(gilrs::Button::RightTrigger2)
                .map(gilrs::ev::state::ButtonData::value)
                .unwrap_or(0.0),
        }
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Buttons {
    pub south: bool,
    pub north: bool,
    pub east: bool,
    pub west: bool,
    pub dpad_up: bool,
    pub dpad_down: bool,
    pub dpad_left: bool,
    pub dpad_right: bool,
    pub start: bool,
    pub select: bool,
    pub left_bumper: bool,
    pub right_bumper: bool,
}

impl Buttons {
    pub fn from_game_pad(input: gilrs::Gamepad<'_>) -> Self {
        Self {
            south: input.is_pressed(gilrs::Button::South),
            north: input.is_pressed(gilrs::Button::North),
            east: input.is_pressed(gilrs::Button::East),
            west: input.is_pressed(gilrs::Button::West),
            start: input.is_pressed(gilrs::Button::Start),
            select: input.is_pressed(gilrs::Button::Select),
            left_bumper: input.is_pressed(gilrs::Button::LeftTrigger),
            right_bumper: input.is_pressed(gilrs::Button::RightTrigger),
            dpad_up: input.is_pressed(gilrs::Button::DPadUp),
            dpad_down: input.is_pressed(gilrs::Button::DPadDown),
            dpad_left: input.is_pressed(gilrs::Button::DPadLeft),
            dpad_right: input.is_pressed(gilrs::Button::DPadRight),
        }
    }
}
