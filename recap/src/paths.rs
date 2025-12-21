/// Paths for the application
#[derive(Debug)]
pub struct Paths {
    pub temp_dir: std::path::PathBuf,
    pub recordings_dir: std::path::PathBuf,
    pub state_file: std::path::PathBuf,
    pub state_dir: std::path::PathBuf,
}

/// Get the paths for the application
pub fn get_paths() -> &'static Paths {
    static INIT: std::sync::OnceLock<Paths> = std::sync::OnceLock::new();
    INIT.get_or_init(|| {
        let temp_dir = std::env::temp_dir().join("com.elefant.recap");
        if !temp_dir.exists() {
            std::fs::create_dir(&temp_dir).unwrap();
        }
        let recordings_dir = temp_dir.join("recordings");
        if !recordings_dir.exists() {
            std::fs::create_dir(&recordings_dir).unwrap();
        }
        #[cfg(not(windows))]
        let home_dir = std::env::var("HOME").map(std::path::PathBuf::from).unwrap();
        #[cfg(windows)]
        let home_dir = std::env::var("USERPROFILE")
            .map(std::path::PathBuf::from)
            .unwrap();

        let state_dir = home_dir.join(".recap");
        if !state_dir.exists() {
            std::fs::create_dir(&state_dir).unwrap();
        }

        let state_file = state_dir.join("state.json");

        Paths {
            state_file,
            state_dir,
            recordings_dir,
            temp_dir,
        }
    })
}

pub fn get_annotation_path(uuid: &uuid::Uuid) -> std::path::PathBuf {
    get_paths()
        .recordings_dir
        .join(uuid.to_string())
        .join("annotation.proto")
}
