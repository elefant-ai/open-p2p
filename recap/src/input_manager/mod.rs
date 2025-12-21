mod double_check;
pub mod game_pad;
pub mod keyboard;
pub mod mouse;
pub mod simulate;
pub mod timeline;
use double_check::double_check_keycode;
use glam::IVec2;
use input_codes::Keycode;
use parking_lot::Mutex;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator as _};
use simulate::simulate_key;
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, LazyLock, atomic::AtomicBool},
};
use tracing::{error, info, trace, warn};
use winit::event::RawKeyEvent;

use crate::{input_manager::timeline::TIMELINE, sound::FileSource};

#[derive(Debug, Clone)]
pub struct DeviceEvent {
    pub time: std::time::SystemTime,
    pub event: Event,
    pub simulated: bool,
}

#[derive(Debug, Clone)]
pub enum Event {
    MouseMove(IVec2),
    MouseDelta(IVec2),
    MouseWheel(IVec2),
    MouseButton {
        pressed: bool,
        button: input_codes::Button,
    },
    KeyboardInput {
        key: input_codes::Keycode,
        pressed: bool,
    },
    GamePadAction(gilrs::EventType),
}

static DEVICEEVENTS: LazyLock<flume::Sender<DeviceEvent>> = LazyLock::new(|| {
    let (sender, receiver) = flume::unbounded();
    handle_device_events(receiver);
    sender
});

fn handle_device_events(recv: flume::Receiver<DeviceEvent>) {
    std::thread::spawn(move || {
        while let Ok(event) = recv.recv() {
            #[cfg(not(feature = "inference"))]
            let event = {
                let mut event = event;
                event.simulated = false;
                event
            };

            let mut state = INPUT_STATE.lock();
            timeline::push_timeline_event(event.clone());
            SEND_TO_LISTENERS.send(event.clone()).unwrap_or_else(|err| {
                error!("Failed to send device event to listeners: {}", err);
            });
            let DeviceEvent {
                event, simulated, ..
            } = event;
            match event {
                Event::MouseMove(ivec2) => {
                    state.handle_event(Event::MouseMove(ivec2), simulated);
                }
                Event::MouseDelta(ivec2) => {
                    state.handle_event(Event::MouseDelta(ivec2), simulated);
                }
                Event::MouseWheel(ivec2) => {
                    state.handle_event(Event::MouseWheel(ivec2), simulated);
                }
                Event::MouseButton { pressed, button } => {
                    state.handle_event(Event::MouseButton { pressed, button }, simulated);
                }
                Event::KeyboardInput { key, pressed } => {
                    state.handle_event(Event::KeyboardInput { key, pressed }, simulated);
                }
                Event::GamePadAction(event_type) => {
                    state.handle_event(Event::GamePadAction(event_type), simulated);
                }
            }
        }
    });
}

fn send_device_event(event: DeviceEvent) {
    if let Err(err) = DEVICEEVENTS.send(event) {
        error!("Failed to send device event: {}", err);
    }
}

struct Listeners {
    listeners: HashMap<u64, Box<dyn FnMut(&DeviceEvent, u64) + Send + Sync>>,
    id: u64,
}

static LISTENERS: LazyLock<Mutex<Listeners>> = LazyLock::new(|| {
    Mutex::new(Listeners {
        listeners: HashMap::new(),
        id: 0,
    })
});

static SEND_TO_LISTENERS: LazyLock<flume::Sender<DeviceEvent>> = LazyLock::new(|| {
    let (sender, receiver) = flume::unbounded();
    handle_listeners(receiver);
    sender
});

/// all hotkeys that are used in the application
pub static HOT_KEYS: LazyLock<HashSet<input_codes::Keycode>> = LazyLock::new(|| {
    let mut hot_keys = HashSet::new();
    crate::hot_key::TOGGLE_RECORDING_HOTKEY
        .iter()
        .for_each(|key| {
            hot_keys.insert(key.clone());
        });
    crate::hot_key::TOGGLE_RECORDING_WITH_INFERENCE_HOTKEY
        .iter()
        .for_each(|key| {
            hot_keys.insert(key.clone());
        });
    crate::hot_key::TOGGLE_MODEL_CONTROL_HOTKEY
        .iter()
        .for_each(|key| {
            hot_keys.insert(key.clone());
        });
    hot_keys
});

/// Setup the key manager
/// WARNING: This function should only be called once in the entire program and must be called from the main thread
pub fn setup() {
    static SETUP: AtomicBool = AtomicBool::new(false);
    if SETUP.swap(true, std::sync::atomic::Ordering::Relaxed) {
        panic!("Key manager already setup");
    }
    handle_rdev_events();
}

fn handle_rdev_events() {
    std::thread::spawn(move || {
        if let Err(err) = rdev::listen(move |event| match event.event_type {
            // TODO: @night-hunter make simulated work here
            rdev::EventType::MouseMove { x, y } => {
                let ivec2 = IVec2::new(x as i32, y as i32);
                send_device_event(DeviceEvent {
                    time: std::time::SystemTime::now(),
                    event: Event::MouseMove(ivec2),
                    simulated: false,
                });
            }
            rdev::EventType::Wheel { delta_x, delta_y } => {
                let ivec = IVec2::new(delta_x as i32, delta_y as i32);
                send_device_event(DeviceEvent {
                    time: std::time::SystemTime::now(),
                    event: Event::MouseWheel(ivec),
                    simulated: false,
                });
            }
            _ => {}
        }) {
            panic!("Error listening for keyboard events: {err:?}");
        };
    });
}

fn handle_listeners(recv: flume::Receiver<DeviceEvent>) {
    // Spawn a thread to listen for events
    std::thread::spawn(move || {
        while let Ok(event) = recv.recv() {
            let mut listeners = LISTENERS.lock();
            listeners
                .listeners
                .par_iter_mut()
                .for_each(|(id, listener)| listener(&event, *id));
        }
    });
}

const SKIP_KEYS: &[input_codes::Keycode] = &[
    input_codes::Keycode::VolumeMute,
    input_codes::Keycode::VolumeUp,
    input_codes::Keycode::VolumeDown,
];

/// take the raw winit device events and process it
#[inline(always)]
pub fn handle_device_event(device_id: winit::event::DeviceId, event: winit::event::DeviceEvent) {
    let time = std::time::SystemTime::now();
    match event {
        winit::event::DeviceEvent::Button { button, state } => {
            let pressed = matches!(state, winit::event::ElementState::Pressed);
            let button = input_codes::Button::from(button as u8);
            send_device_event(DeviceEvent {
                time,
                event: Event::MouseButton { pressed, button },
                simulated: device_id == winit::event::DeviceId::dummy(),
            });
        }
        // winit::event::DeviceEvent::MouseWheel { delta } => {}
        winit::event::DeviceEvent::MouseMotion { delta } => {
            if delta.0.fract() != 0.0 || delta.1.fract() != 0.0 {
                tracing::warn!("Mouse delta is not an integer: {:?}", delta);
            }
            let ivec = IVec2::new(delta.0 as i32, delta.1 as i32);
            send_device_event(DeviceEvent {
                time,
                event: Event::MouseDelta(ivec),
                simulated: device_id == winit::event::DeviceId::dummy(),
            });
        }
        winit::event::DeviceEvent::Key(RawKeyEvent {
            state,
            physical_key,
        }) => {
            let pressed = matches!(state, winit::event::ElementState::Pressed);
            match physical_key {
                winit::keyboard::PhysicalKey::Code(key_code) => {
                    let key = input_codes::Keycode::from(key_code);
                    if SKIP_KEYS.contains(&key) {
                        return;
                    }
                    send_device_event(DeviceEvent {
                        time,
                        event: Event::KeyboardInput { key, pressed },
                        simulated: device_id == winit::event::DeviceId::dummy(),
                    });
                }
                winit::keyboard::PhysicalKey::Unidentified(native_key_code) => {
                    tracing::warn!("Unidentified key code: {:?}", native_key_code);
                }
            }
        }
        _ => {}
    }
}

pub fn reset_recording() -> std::time::SystemTime {
    let mut state = INPUT_STATE.lock();
    state.reset();
    timeline::start_timeline()
}

#[inline(always)]
/// Collect the currently pressed keys and simulated keys
/// All State is collected from the INPUT_LISTENER state to hold the lock for the shortest time possible while collecting all data to avoid events processing while collecting the state
pub fn collect_input_frames() -> crate::handler::capture::InputFrame {
    let mut state = INPUT_STATE.lock();
    let mut timeline_guard = TIMELINE.lock();
    let inference_running = state
        .inference_running
        .as_ref()
        .map(|arc| arc.load(std::sync::atomic::Ordering::Relaxed))
        .unwrap_or(false);
    let timeline = timeline_guard.drain_frame_events();
    let time = std::time::SystemTime::now();

    let delta = std::mem::take(&mut state.mouse_delta);
    let scroll = std::mem::take(&mut state.scroll_delta);
    let system_delta = std::mem::take(&mut state.simulated_mouse_delta);
    let system_scroll = std::mem::take(&mut state.simulated_scroll_delta);
    let system_mouse_pos = std::mem::take(&mut state.simulated_mouse_position);

    let buttons_set = state.currently_pressed_mouse_buttons.clone();
    let mouse_pos = state.current_mouse_position;

    let user_keys = state
        .currently_pressed_keys
        .keys()
        .cloned()
        .collect::<Vec<_>>();

    let simulated_keys = state.simulated_key.clone();
    let simulated_mouse = state.simulated_mouse_buttons.clone();

    let game_pad = game_pad::get_state();

    // Before any processing drop the state to avoid holding the lock
    drop(state);
    drop(timeline_guard);

    crate::handler::capture::InputFrame {
        time,
        user_mouse: crate::handler::capture::InputFrameMouse {
            delta: delta.into_iter().sum(),
            mouse_pos,
            buttons: buttons_set.into_iter().collect(),
            scroll: scroll.into_iter().sum(),
        },
        system_mouse: crate::handler::capture::InputFrameMouse {
            delta: system_delta.into_iter().sum(),
            mouse_pos: system_mouse_pos,
            buttons: simulated_mouse.into_iter().collect(),
            scroll: system_scroll.into_iter().sum(),
        },
        user_keys: user_keys
            .into_iter()
            .filter(|k| !HOT_KEYS.contains(k))
            .collect(),
        system_keys: simulated_keys
            .into_iter()
            .filter(|k| !HOT_KEYS.contains(k))
            .collect(),
        inference_running,
        game_pad,
        timeline,
    }
}

// Checks the current state of each key code by making a system call to check if the key is pressed then updating the state to match the system state
pub fn double_check_key_state() {
    let mut hash_set = HashMap::new();
    // Lock the state while we get the keyboard situation to prevent any event-driven changes happening in the meantime.
    let mut state = INPUT_STATE.lock();
    input_codes::Keycode::iter()
        .filter(|key| double_check_keycode(key.clone()).unwrap_or(false))
        .for_each(|key| {
            hash_set.insert(key, std::time::Instant::now());
        });

    if !hash_set.is_empty() {
        warn!(
            module = "capture::input_manager",
            "Starting a record with keys down: {:?}", hash_set
        );
    }
    state.currently_pressed_keys = hash_set;
}

/// Lift all simulated keys
/// only removing ones that werent pressed by the user
pub fn lift_simulated_keys() {
    let mut state = INPUT_STATE.lock();
    state.lift_simulated_keys_inner(false);
}

/// Listen for keyboard events
/// this is not a blocking call
pub fn listen<F>(listener: F) -> u64
where
    F: FnMut(&DeviceEvent, u64) + Send + Sync + 'static,
{
    let mut listeners = LISTENERS.lock();
    let id = listeners.id;
    listeners.listeners.insert(id, Box::new(listener));
    listeners.id += 1;
    id
}

/// Remove a listener by its id
pub fn remove_listener(id: u64) {
    let mut listeners = LISTENERS.lock();
    listeners.listeners.remove(&id);
}

static INPUT_STATE: LazyLock<Mutex<InputState>> = LazyLock::new(|| Mutex::new(InputState::new()));

pub fn read_input_state<F, O>(read: F) -> O
where
    F: FnOnce(&InputState) -> O,
{
    let state = INPUT_STATE.lock();
    read(&state)
}

pub fn set_inference_running(inference_running: Option<Arc<AtomicBool>>) {
    let mut state = INPUT_STATE.lock();
    state.inference_running = inference_running;
}

#[derive(derive_more::Debug)]
pub struct InputState {
    pub currently_pressed_keys: HashMap<input_codes::Keycode, std::time::Instant>,
    pub currently_pressed_mouse_buttons: HashSet<input_codes::Button>,
    pub current_mouse_position: IVec2,
    pub mouse_delta: Vec<IVec2>,
    pub scroll_delta: Vec<IVec2>,
    pub simulated_key: HashSet<input_codes::Keycode>,
    pub simulated_mouse_buttons: HashSet<input_codes::Button>,
    pub simulated_mouse_position: IVec2,
    pub simulated_scroll_delta: Vec<IVec2>,
    pub simulated_mouse_delta: Vec<IVec2>,
    pub inference_running: Option<Arc<AtomicBool>>,
}

impl InputState {
    fn new() -> Self {
        Self {
            currently_pressed_keys: HashMap::with_capacity(10),
            currently_pressed_mouse_buttons: HashSet::with_capacity(10),
            current_mouse_position: IVec2::new(0, 0),
            mouse_delta: Vec::with_capacity(50),
            scroll_delta: Vec::with_capacity(50),
            simulated_key: HashSet::with_capacity(10),
            simulated_mouse_buttons: HashSet::with_capacity(10),
            simulated_mouse_position: IVec2::new(0, 0),
            simulated_scroll_delta: Vec::with_capacity(50),
            simulated_mouse_delta: Vec::with_capacity(50),
            inference_running: None,
        }
    }

    fn handle_inference_stop(&mut self, event: &Event, simulated: bool) {
        if simulated {
            return;
        }
        let Some(inference_running) = &self.inference_running else {
            return;
        };
        match &event {
            Event::KeyboardInput { pressed: true, .. }
            | Event::MouseButton { pressed: true, .. } => {
                // if the user input is currently false, set it to true
                if inference_running
                    .compare_exchange(
                        true,
                        false,
                        std::sync::atomic::Ordering::Relaxed,
                        std::sync::atomic::Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    info!("Stopping model control - user input detected {:?}", event);
                    self.lift_simulated_keys_inner(true);
                    FileSource::ModelControlStopped.play();
                } else if let Event::KeyboardInput {
                    pressed: true,
                    key: Keycode::LeftBracket,
                } = &event
                    && inference_running
                        .compare_exchange(
                            false,
                            true,
                            std::sync::atomic::Ordering::Relaxed,
                            std::sync::atomic::Ordering::Relaxed,
                        )
                        .is_ok()
                {
                    info!("Starting model control - user input detected {:?}", event);
                    FileSource::ModelControlStarted.play();
                }
            }
            _ => {}
        }
    }

    fn handle_event(&mut self, event: Event, simulated: bool) {
        self.handle_inference_stop(&event, simulated);
        match event {
            Event::MouseButton { pressed, button } => {
                if pressed {
                    // If the button is simulated, ignore it
                    if simulated {
                        self.simulated_mouse_buttons.insert(button);
                    } else {
                        self.currently_pressed_mouse_buttons.insert(button);
                    }
                } else if simulated {
                    self.simulated_mouse_buttons.remove(&button);
                } else {
                    self.currently_pressed_mouse_buttons.remove(&button);
                }
            }
            Event::KeyboardInput { pressed, key } => {
                if pressed {
                    if simulated {
                        self.simulated_key.insert(key);
                    } else {
                        self.currently_pressed_keys
                            .entry(key)
                            .or_insert_with(std::time::Instant::now);
                    }
                } else if simulated {
                    self.simulated_key.remove(&key);
                } else if let Some(time) = self.currently_pressed_keys.remove(&key) {
                    let duration = time.elapsed();
                    trace!("Key {:?} was pressed for {:#?}", key, duration);
                }
            }
            Event::MouseMove(position) => {
                if simulated {
                    self.simulated_mouse_position = position;
                } else {
                    self.current_mouse_position = position;
                }
            }
            Event::MouseWheel(delta) => {
                if simulated {
                    self.simulated_scroll_delta.push(delta);
                } else {
                    self.scroll_delta.push(delta);
                }
            }
            Event::MouseDelta(ivec2) => {
                if simulated {
                    self.simulated_mouse_delta.push(ivec2);
                } else {
                    self.mouse_delta.push(ivec2);
                }
            }
            Event::GamePadAction(_) => {}
        }
    }

    /// if skip wait is true the function will not wait for the key to be released to remove it from the simulated keys
    fn lift_simulated_keys_inner(&mut self, skip_wait: bool) {
        self.simulated_key.clone().into_iter().for_each(|key| {
            if self.currently_pressed_keys.contains_key(&key) {
                self.simulated_key.remove(&key);
            } else {
                if skip_wait {
                    self.simulated_key.remove(&key);
                }
                simulate_key(key, false);
            }
        });

        self.simulated_mouse_buttons
            .clone()
            .into_iter()
            .for_each(|button| {
                if self.currently_pressed_mouse_buttons.contains(&button) {
                    self.simulated_mouse_buttons.remove(&button);
                } else {
                    if skip_wait {
                        self.simulated_mouse_buttons.remove(&button);
                    }
                    simulate::simulate_mouse_button(button, false);
                }
            });
    }

    fn reset(&mut self) {
        self.mouse_delta.clear();
        self.scroll_delta.clear();
        self.simulated_scroll_delta.clear();
        self.simulated_mouse_delta.clear();
        self.simulated_key.clear();
        self.simulated_mouse_buttons.clear();
        self.simulated_mouse_position = IVec2::new(0, 0);
        self.currently_pressed_keys.clear();
        self.currently_pressed_mouse_buttons.clear();
    }
}
