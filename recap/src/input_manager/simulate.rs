use glam::{DVec2, IVec2};
use tracing::error;

pub use super::mouse::simulate_mouse_delta;

/// Simulate a key press
pub fn simulate_key(key: input_codes::Keycode, press: bool) {
    if let Ok(key) = key
        .try_into()
        .inspect_err(|e| error!("Error converting key: {:?}", e))
    {
        let event = if press {
            rdev::EventType::KeyPress(key)
        } else {
            rdev::EventType::KeyRelease(key)
        };
        let _ =
            rdev::simulate(&event).inspect_err(|e| error!("Error simulating key release: {:?}", e));
    }
}

pub fn simulate_mouse_button(button: input_codes::Button, press: bool) {
    let event = if press {
        rdev::EventType::ButtonPress(button.into())
    } else {
        rdev::EventType::ButtonRelease(button.into())
    };
    let _ =
        rdev::simulate(&event).inspect_err(|e| error!("Error simulating mouse button: {:?}", e));
}

pub fn simulate_mouse_absolute(delta: DVec2) {
    let event = rdev::EventType::MouseMove {
        x: delta.x,
        y: delta.y,
    };
    let _ = rdev::simulate(&event).inspect_err(|e| error!("Error simulating mouse delta: {:?}", e));
}

pub fn simulate_mouse_scroll(delta: IVec2) {
    if delta.x == 0 && delta.y == 0 {
        return;
    }
    let event = rdev::EventType::Wheel {
        delta_x: delta.x as i64,
        delta_y: delta.y as i64,
    };
    let _ =
        rdev::simulate(&event).inspect_err(|e| error!("Error simulating mouse scroll: {:?}", e));
}
