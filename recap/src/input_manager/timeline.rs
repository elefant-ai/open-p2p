use std::sync::LazyLock;

use parking_lot::Mutex;

pub static TIMELINE: LazyLock<Mutex<Timeline>> = LazyLock::new(|| Mutex::new(Timeline::default()));

/// Start a timeline to collect events
pub fn start_timeline() -> std::time::SystemTime {
    let mut timeline = TIMELINE.lock();
    timeline.start = std::time::SystemTime::now();
    timeline.events.clear();
    timeline.full_events.clear();
    timeline.start
}

pub fn push_timeline_event(event: super::DeviceEvent) {
    let mut timeline = TIMELINE.lock();
    timeline.push_event(event);
}

#[derive(Debug, Clone)]
pub struct Timeline {
    pub start: std::time::SystemTime,
    pub events: Vec<super::DeviceEvent>,
    pub full_events: Vec<super::DeviceEvent>,
}

impl Default for Timeline {
    fn default() -> Self {
        Self {
            start: std::time::SystemTime::now(),
            events: Vec::new(),
            full_events: Vec::new(),
        }
    }
}

impl Timeline {
    pub fn drain_frame_events(&mut self) -> Vec<super::DeviceEvent> {
        std::mem::take(&mut self.events)
    }

    pub fn drain_full_events(&mut self) -> Vec<super::DeviceEvent> {
        std::mem::take(&mut self.full_events)
    }

    pub fn push_event(&mut self, event: super::DeviceEvent) {
        self.events.push(event.clone());
        self.full_events.push(event);
    }
}
