use winit::raw_window_handle::HasWindowHandle;

#[derive(Debug, Clone)]
pub struct InnerWindow {
    pub title: String,
    pub window: window_handling::Window,
}

impl PartialEq for InnerWindow {
    fn eq(&self, other: &Self) -> bool {
        let hwnd = self.window.window_handle().unwrap();
        let other_hwnd = other.window.window_handle().unwrap();
        hwnd == other_hwnd
    }
}

impl InnerWindow {
    pub fn new(title: String, window: window_handling::Window) -> Self {
        Self { title, window }
    }
}

impl std::fmt::Display for InnerWindow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.title)
    }
}
