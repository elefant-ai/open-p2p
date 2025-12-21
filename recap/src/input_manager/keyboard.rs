pub mod layout;

/// Get the current keyboard layout name.
pub fn keyboard_layout() -> Result<layout::KeyboardLayout, anyhow::Error> {
    let mut buf = [0u16; 9];
    #[allow(unsafe_code)]
    unsafe { windows::Win32::UI::Input::KeyboardAndMouse::GetKeyboardLayoutNameW(&mut buf) }?;
    let layout = String::from_utf16_lossy(&buf).trim().to_string();
    if layout.is_empty() {
        return Err(anyhow::anyhow!("Keyboard layout is empty"));
    }
    let without_prefix = layout
        .trim_start_matches("0x")
        .trim_end_matches("\0")
        .trim();
    println!("Keyboard layout: {without_prefix:?}");
    let num = u32::from_str_radix(without_prefix, 16).map_err(|err| {
        println!("Failed to parse keyboard layout number: {err}");
        anyhow::anyhow!("Failed to parse keyboard layout number: {without_prefix}")
    })?;

    let layout = layout::KeyboardLayout::try_from(num)
        .map_err(|_| anyhow::anyhow!("Failed to convert keyboard layout: {layout}"))?;
    Ok(layout)
}
