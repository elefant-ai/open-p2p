#![cfg(target_os = "windows")]
pub use windows::double_check_keycode;

#[cfg(target_os = "windows")]
mod windows {
    use windows::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;

    /// Check if a key is pressed using the Windows API
    pub fn double_check_keycode(keycode: input_codes::Keycode) -> Result<bool, anyhow::Error> {
        #[allow(unsafe_code)]
        let res = unsafe {
            GetAsyncKeyState(
                rdev::win_code_from_key(keycode.try_into().map_err(|err| {
                    anyhow::anyhow!("Failed to convert keycode to Windows code: {err}")
                })?)
                .ok_or_else(|| anyhow::anyhow!("Unabled to convert keycode"))?
                    as i32,
            ) as u32
        };

        if res & 0x8000 != 0 {
            Ok(true)
        } else {
            Ok(false)
        }
    }
}
