use iced::{
    Element,
    widget::{button, column, container, row, text, text_input},
};
use window_handling::WindowInfo as _;

use crate::{App, Message};

#[cfg(target_os = "windows")]
use windows::Win32::UI::WindowsAndMessaging::{SWP_NOMOVE, SWP_NOSIZE, SWP_NOZORDER, SetWindowPos};

#[cfg(target_os = "windows")]
use window_handling::raw_window_handle;

#[derive(Debug, Clone, Default)]
pub struct CachedWindowInfo {
    pub size: Option<(i32, i32)>,
    pub position: Option<(i32, i32)>,
    pub scale_factor: Option<f64>,
}

#[derive(Debug, Clone)]
pub enum WindowSizeMessage {
    RefreshSize,
    SetWidth(i32),
    SetHeight(i32),
    ApplySize,
    SetPresetSize(i32, i32),
    SetX(i32),
    SetY(i32),
    ApplyPosition,
    SetPresetPosition(i32, i32),
}

const MIN_WINDOW_WIDTH: i32 = 100;
const MAX_WINDOW_WIDTH: i32 = 7680;
const MIN_WINDOW_HEIGHT: i32 = 100;
const MAX_WINDOW_HEIGHT: i32 = 4320;

pub(crate) fn window_size_display(state: &App) -> Element<'_, Message> {
    if state.target.is_some() {
        let size_info = match state.cached_window_info.size {
            Some((width, height)) => {
                format!("Current size: {width}x{height} pixels")
            }
            None => "Click 'Refresh Size Info' to get window size".to_string(),
        };

        let position_info = match state.cached_window_info.position {
            Some((x, y)) => {
                format!("Position: ({x}, {y})")
            }
            None => "Click 'Refresh Size Info' to get window position".to_string(),
        };

        let scale_info = match state.cached_window_info.scale_factor {
            Some(scale_factor) => format!("Scale factor: {scale_factor:.2}"),
            None => "Click 'Refresh Size Info' to get scale factor".to_string(),
        };

        let refresh_button = button("Refresh Size Info")
            .on_press(Message::WindowSize(WindowSizeMessage::RefreshSize));
        let info_column = column![
            text("Window Information").size(16),
            text(size_info),
            text(position_info),
            text(scale_info),
            refresh_button,
        ]
        .spacing(5);
        container(info_column)
            .padding(10)
            .style(|theme: &iced::Theme| {
                iced::widget::container::Style::default()
                    .border(iced::border::rounded(5))
                    .background(theme.palette().primary)
            })
            .into()
    } else {
        text("No target selected").into()
    }
}

pub(crate) fn window_size_control(state: &App) -> Element<'_, Message> {
    if state.target.is_some() {
        let width_input = text_input("Width", &state.saved_state.target_width.to_string())
            .on_input(|s| {
                Message::WindowSize(WindowSizeMessage::SetWidth(s.parse().unwrap_or(600)))
            })
            .width(50);

        let height_input = text_input("Height", &state.saved_state.target_height.to_string())
            .on_input(|s| {
                Message::WindowSize(WindowSizeMessage::SetHeight(s.parse().unwrap_or(400)))
            })
            .width(50);

        let apply_button =
            button("Apply").on_press(Message::WindowSize(WindowSizeMessage::ApplySize));

        let control_row = row![
            text("Set size:"),
            width_input,
            text("x"),
            height_input,
            apply_button,
        ]
        .spacing(10);

        // Add preset size buttons
        let preset_row = row![
            text("Presets:"),
            button("720p").on_press(Message::WindowSize(WindowSizeMessage::SetPresetSize(
                1280, 720
            ))),
            button("1080p").on_press(Message::WindowSize(WindowSizeMessage::SetPresetSize(
                1920, 1080
            ))),
            button("800x600").on_press(Message::WindowSize(WindowSizeMessage::SetPresetSize(
                800, 600
            ))),
        ]
        .spacing(3);

        // Add position controls
        let x_input = text_input("X", &state.saved_state.target_x.to_string())
            .on_input(|s| Message::WindowSize(WindowSizeMessage::SetX(s.parse().unwrap_or(100))))
            .width(50);

        let y_input = text_input("Y", &state.saved_state.target_y.to_string())
            .on_input(|s| Message::WindowSize(WindowSizeMessage::SetY(s.parse().unwrap_or(100))))
            .width(50);

        let apply_position_button =
            button("Apply").on_press(Message::WindowSize(WindowSizeMessage::ApplyPosition));

        let position_row = row![
            text("Set position:"),
            x_input,
            text(","),
            y_input,
            apply_position_button,
        ]
        .spacing(10);

        // Add preset position buttons
        let position_preset_row = row![
            text("Position presets:"),
            button("Top-Left").on_press(Message::WindowSize(WindowSizeMessage::SetPresetPosition(
                100, 100
            ))),
            button("Center").on_press(Message::WindowSize(WindowSizeMessage::SetPresetPosition(
                400, 300
            ))),
            button("Top-Right").on_press(Message::WindowSize(
                WindowSizeMessage::SetPresetPosition(800, 100)
            )),
        ]
        .spacing(3);

        let full_control =
            column![control_row, preset_row, position_row, position_preset_row].spacing(10);

        container(full_control)
            .padding(10)
            .style(|theme: &iced::Theme| {
                iced::widget::container::Style::default()
                    .border(iced::border::rounded(5))
                    .background(theme.palette().primary)
            })
            .into()
    } else {
        text("Select a target to control window size").into()
    }
}

pub(crate) fn update_window_size(
    state: &mut App,
    message: WindowSizeMessage,
) -> iced::Task<Message> {
    match message {
        WindowSizeMessage::RefreshSize => {
            if let Some(target) = &state.target {
                state.cached_window_info.size = target.window.size().ok();
                state.cached_window_info.position = target.window.position().ok();
                state.cached_window_info.scale_factor = Some(target.window.scale_factor());
            }
        }
        WindowSizeMessage::SetWidth(width) => {
            state.saved_state.target_width = width;
        }
        WindowSizeMessage::SetHeight(height) => {
            state.saved_state.target_height = height;
        }
        WindowSizeMessage::SetPresetSize(width, height) => {
            state.saved_state.target_width = width;
            state.saved_state.target_height = height;
        }
        WindowSizeMessage::ApplySize => {
            if let Some(target) = &state.target {
                let width = state.saved_state.target_width;
                let height = state.saved_state.target_height;

                // Validate window size ranges for safety
                if !(MIN_WINDOW_WIDTH..=MAX_WINDOW_WIDTH).contains(&width) {
                    state.error = Some(format!(
                        "Width must be between {MIN_WINDOW_WIDTH} and {MAX_WINDOW_WIDTH} pixels, got {width}"
                    ));
                    return iced::Task::none();
                }
                if !(MIN_WINDOW_HEIGHT..=MAX_WINDOW_HEIGHT).contains(&height) {
                    state.error = Some(format!(
                        "Height must be between {MIN_WINDOW_HEIGHT} and {MAX_WINDOW_HEIGHT} pixels, got {height}"
                    ));
                    return iced::Task::none();
                }

                match resize_window(&target.window, width, height) {
                    Ok(()) => {
                        state.error = None;
                        // Eagerly refresh window info after successful resize
                        state.cached_window_info.size = target.window.size().ok();
                        state.cached_window_info.position = target.window.position().ok();
                        state.cached_window_info.scale_factor = Some(target.window.scale_factor());
                    }
                    Err(e) => {
                        state.error = Some(format!("Failed to resize window: {e}"));
                    }
                }
            } else {
                state.error = Some("No target window selected".to_string());
            }
        }
        WindowSizeMessage::SetX(x) => {
            state.saved_state.target_x = x;
        }
        WindowSizeMessage::SetY(y) => {
            state.saved_state.target_y = y;
        }
        WindowSizeMessage::SetPresetPosition(x, y) => {
            state.saved_state.target_x = x;
            state.saved_state.target_y = y;
        }
        WindowSizeMessage::ApplyPosition => {
            if let Some(target) = &state.target {
                let x = state.saved_state.target_x;
                let y = state.saved_state.target_y;

                match move_window(target, x, y) {
                    Ok(()) => {
                        state.error = None;
                    }
                    Err(e) => {
                        state.error = Some(format!("Failed to move window: {e}"));
                    }
                }
            } else {
                state.error = Some("No target window selected".to_string());
            }
        }
    };
    iced::Task::none()
}

#[cfg(target_os = "windows")]
fn resize_window<W: window_handling::WindowInfo + Clone>(
    target: &W,
    width: i32,
    height: i32,
) -> Result<(), String> {
    // Check if the window is minimized or invalid
    if target.minimized() {
        return Err(
            "Cannot resize a minimized window. Please restore the window first.".to_string(),
        );
    }

    // Get the window handle from the target
    let window_handle = target
        .window_handle()
        .map_err(|e| format!("Failed to get window handle: {e}"))?;

    // Extract the HWND from the raw window handle
    if let raw_window_handle::RawWindowHandle::Win32(win32_handle) = window_handle.as_raw() {
        let hwnd = windows::Win32::Foundation::HWND(win32_handle.hwnd.get() as _);

        // Use SetWindowPos to resize the window without moving it
        #[allow(unsafe_code)]
        unsafe {
            SetWindowPos(
                hwnd,
                None, // hWndInsertAfter - don't change Z-order
                0,    // X - don't change position
                0,    // Y - don't change position
                width,
                height,
                SWP_NOMOVE | SWP_NOZORDER, // Don't move or change Z-order
            )
            .map_err(|e| format!("Windows API SetWindowPos failed: {e}"))?;
        }

        Ok(())
    } else {
        Err("Window handle is not a Win32 handle. This feature only works on Windows.".to_string())
    }
}

#[cfg(not(target_os = "windows"))]
fn resize_window<W: window_handling::WindowInfo + Clone>(
    _target: &W,
    width: i32,
    height: i32,
) -> Result<(), String> {
    Err(format!(
        "Window resizing not implemented for this platform. Would resize to {}x{}",
        width, height
    ))
}

#[cfg(target_os = "windows")]
fn move_window(target: &crate::InnerWindow, x: i32, y: i32) -> Result<(), String> {
    // Check if the window is minimized or invalid

    use winit::raw_window_handle::HasWindowHandle;
    if target.window.minimized() {
        return Err("Cannot move a minimized window. Please restore the window first.".to_string());
    }

    // Get the window handle from the target
    let window_handle = target
        .window
        .window_handle()
        .map_err(|e| format!("Failed to get window handle: {e:?}"))?;

    if let raw_window_handle::RawWindowHandle::Win32(win32_handle) = window_handle.as_raw() {
        let hwnd = windows::Win32::Foundation::HWND(win32_handle.hwnd.get() as _);

        // Get current size to preserve it
        let (current_width, current_height) = target
            .window
            .size()
            .map_err(|e| format!("Failed to get current window size: {e:?}"))?;

        // Use SetWindowPos to move the window without changing size
        #[allow(unsafe_code)]
        unsafe {
            SetWindowPos(
                hwnd,
                None, // hWndInsertAfter - don't change Z-order
                x,
                y,
                current_width,
                current_height,
                SWP_NOSIZE | SWP_NOZORDER, // Don't resize or change Z-order
            )
        }
        .map_err(|e| format!("Windows API SetWindowPos failed: {e}"))?;

        Ok(())
    } else {
        Err("Window handle is not a Win32 handle. This feature only works on Windows.".to_string())
    }
}

#[cfg(not(target_os = "windows"))]
fn move_window<W: window_handling::WindowInfo + Clone>(
    _target: &crate::InnerWindow<W>,
    x: i32,
    y: i32,
) -> Result<(), String> {
    Err(format!(
        "Window moving not implemented for this platform. Would move to ({}, {})",
        x, y
    ))
}
