//! Helper functions for the capture module
use std::str::FromStr as _;

use input_codes::{Button, Keycode};
use rayon::iter::{IntoParallelRefIterator as _, ParallelIterator as _};

use crate::input_manager::{
    mouse::simulate_mouse_delta,
    simulate::{
        simulate_key, simulate_mouse_absolute, simulate_mouse_button, simulate_mouse_scroll,
    },
};

// // watch for no activity and set the user input to false
// pub async fn watch_for_no_activity(
//     stop: Arc<AtomicBool>,
//     user_input: Arc<AtomicBool>,
//     last_user_input_time: Arc<std::sync::Mutex<std::time::Instant>>,
//     time_to_wait: std::time::Duration,
// ) {
//     loop {
//         if stop.load(std::sync::atomic::Ordering::SeqCst) {
//             break;
//         }
//         {
//             let mut last_user_input_time = last_user_input_time.lock().unwrap();
//             if last_user_input_time.elapsed() > time_to_wait {
//                 let keys = get_pressed_keys();
//                 trace!("No activity detected {:?}", keys);
//                 if keys.is_empty() {
//                     user_input.store(false, std::sync::atomic::Ordering::Relaxed);
//                 } else {
//                     *last_user_input_time = std::time::Instant::now();
//                 }
//             }
//         }

//         tokio::time::sleep(std::time::Duration::from_millis(500)).await;
//     }
// }
pub fn process_mouse(
    actions: video_inference_grpc::video_inference::MouseAction,
    previous_mouse_buttons: &mut Vec<String>,
) {
    let buttons_down = actions.buttons_down;
    process_mouse_buttons(buttons_down, previous_mouse_buttons);
    let mouse_delta_px = actions.mouse_change;
    if let Some(mouse_delta_px) = mouse_delta_px {
        match mouse_delta_px {
            video_inference_grpc::video_inference::mouse_action::MouseChange::MouseDeltaPx(
                vec2_int,
            ) => {
                simulate_mouse_delta(vec2_int.into());
            }
            video_inference_grpc::video_inference::mouse_action::MouseChange::MousePos(
                vec2_float,
            ) => {
                simulate_mouse_absolute(vec2_float.into());
            }
        }
    }
    let scroll_delta_px = actions.scroll_delta_px;
    if let Some(scroll_delta_px) = scroll_delta_px {
        simulate_mouse_scroll(scroll_delta_px.into());
    }
}

fn process_mouse_buttons(mouse_buttons: Vec<String>, previous_mouse_buttons: &mut Vec<String>) {
    // if the mouse button is not in the previous buttons, simulate a button press
    mouse_buttons
        .par_iter()
        .filter(|button| !previous_mouse_buttons.contains(button))
        .for_each(|button| {
            if let Ok(button) = Button::from_str(button) {
                simulate_mouse_button(button, true);
            }
        });

    // if the mouse button is not in the current buttons, simulate a button release
    previous_mouse_buttons
        .par_iter()
        .filter(|button| !mouse_buttons.contains(button))
        .for_each(|button| {
            if let Ok(button) = Button::from_str(button) {
                simulate_mouse_button(button, false);
            }
        });

    // update the previous mouse buttons to the current mouse buttons
    *previous_mouse_buttons = mouse_buttons;
}

// process the keys received from the server
pub fn process_keys(keys: Vec<String>, previous_keys: &mut Vec<String>) {
    // if the key is not in the previous keys, simulate a key press
    keys.par_iter()
        .filter(|key| !previous_keys.contains(key))
        .for_each(|key| {
            if let Ok(key) = Keycode::from_str(key) {
                simulate_key(key, true);
            }
        });

    // if the key is not in the current keys, simulate a key release
    previous_keys
        .par_iter()
        .filter(|key| !keys.contains(key))
        .for_each(|key| {
            if let Ok(key) = Keycode::from_str(key) {
                simulate_key(key, false);
            }
        });

    // update the previous keys to the current keys
    *previous_keys = keys;
}
