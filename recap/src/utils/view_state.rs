use iced::widget::{Column, text};

use crate::input_manager::read_input_state;

#[derive(Debug, Clone)]
pub struct KeyView {
    pub user_keys: Vec<String>,
    pub user_buttons: Vec<String>,
    pub system_keys: Vec<String>,
    pub system_buttons: Vec<String>,
}

impl Default for KeyView {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyView {
    pub fn new() -> Self {
        Self {
            user_keys: Vec::new(),
            user_buttons: Vec::new(),
            system_keys: Vec::new(),
            system_buttons: Vec::new(),
        }
    }

    pub fn update(&mut self) {
        let (user_keys, user_buttons, system_keys, system_buttons) = read_input_state(|state| {
            (
                state
                    .currently_pressed_keys
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>(),
                state.currently_pressed_mouse_buttons.clone(),
                state.simulated_key.clone(),
                state.simulated_mouse_buttons.clone(),
            )
        });

        self.user_keys = user_keys.into_iter().map(|k| k.to_string()).collect();
        self.user_buttons = user_buttons.into_iter().map(|b| b.to_string()).collect();
        self.system_keys = system_keys.into_iter().map(|k| k.to_string()).collect();
        self.system_buttons = system_buttons.into_iter().map(|b| b.to_string()).collect();
    }

    pub fn view(&self) -> iced::Element<'_, crate::Message> {
        Column::new()
            .push(text("User Keys"))
            .push(text(format!("{:?}", self.user_keys)))
            .push(text("User Buttons"))
            .push(text(format!("{:?}", self.user_buttons)))
            .push(text("System Keys"))
            .push(text(format!("{:?}", self.system_keys)))
            .push(text("System Buttons"))
            .push(text(format!("{:?}", self.system_buttons)))
            .into()
    }
}
