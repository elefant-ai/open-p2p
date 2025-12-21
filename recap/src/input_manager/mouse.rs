use glam::IVec2;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    INPUT, INPUT_0, INPUT_MOUSE, MOUSEEVENTF_MOVE, MOUSEINPUT, SendInput,
};

pub fn simulate_mouse_delta(delta: IVec2) {
    let mut input = INPUT_0::default();

    input.mi = MOUSEINPUT {
        dx: delta.x,
        dy: delta.y,
        mouseData: 0,
        dwFlags: MOUSEEVENTF_MOVE,
        time: 0,
        dwExtraInfo: 0,
    };

    #[allow(unsafe_code)]
    unsafe {
        SendInput(
            &[INPUT {
                r#type: INPUT_MOUSE,
                Anonymous: input,
            }],
            size_of::<INPUT>() as i32,
        )
    };
}
