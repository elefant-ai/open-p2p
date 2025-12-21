use crate::{App, utils::windows::InnerWindow};

/// Snapshot of the application state that can be safely sent across threads
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    pub devices: Vec<InnerWindow>,
    pub target: Option<InnerWindow>,
    pub recording: bool,
    pub env: String,
    pub env_subtype: String,
    pub user: String,
    pub task: String,
    pub current_uuid: Option<uuid::Uuid>,
}

impl From<&mut App> for StateSnapshot {
    fn from(state: &mut App) -> Self {
        Self {
            devices: state.devices.clone(),
            target: state.target.clone(),
            recording: state.handler.running,
            env: state.saved_state.env.clone(),
            env_subtype: state.saved_state.env_subtype.clone(),
            user: state.saved_state.user.clone(),
            task: state.saved_state.task.clone(),
            current_uuid: state.current_uuid,
        }
    }
}
