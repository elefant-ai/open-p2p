use iced::{Element, widget::pick_list};

pub(crate) fn set_target(state: &crate::App) -> Element<'_, crate::Message> {
    pick_list(
        state.devices.clone(),
        state.target.clone(),
        crate::Message::SetTarget,
    )
    .width(200.0)
    .placeholder("Select a target")
    .into()
}

pub(crate) fn set_mic_target(state: &crate::App) -> Element<'_, crate::Message> {
    pick_list(
        state.mic_devices.clone(),
        state.mic.clone(),
        crate::Message::SetMic,
    )
    .width(200.0)
    .placeholder("Select a microphone")
    .into()
}
