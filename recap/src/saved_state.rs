use crate::pages::Pages;

#[derive(Debug, serde::Deserialize, serde::Serialize, Default)]
#[serde(default)]
pub struct SavedState {
    pub env: String,
    pub env_subtype: String,
    pub user: String,
    pub task: String,
    pub target: Option<String>,
    #[serde(default)]
    pub mic: Option<String>,
    #[serde(default = "default_float")]
    pub mic_volume: f64,
    pub recent_env: Vec<String>,
    pub recent_env_subtype: Vec<String>,
    #[serde(default)]
    pub enable_mic_audio: bool,
    #[serde(default)]
    pub page: Pages,
    #[serde(default = "default_value::<800>")]
    pub target_width: i32,
    #[serde(default = "default_value::<600>")]
    pub target_height: i32,
    #[serde(default = "default_value::<100>")]
    pub target_x: i32,
    #[serde(default = "default_value::<100>")]
    pub target_y: i32,
}

fn default_float() -> f64 {
    1.0
}

fn default_value<const N: i32>() -> i32 {
    N
}
