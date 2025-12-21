use iced::{
    Element, Task,
    widget::{self, container, text},
};

use crate::widgets::meta_data::{GIT_COMMIT, RECAP_VERSION};

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum PageMessage {
    SetPage(Pages),
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Pages {
    #[default]
    Home,
}

impl Pages {
    #[allow(dead_code)]
    pub fn options() -> Vec<Pages> {
        vec![Pages::Home]
    }
}

impl From<String> for Pages {
    fn from(_route: String) -> Self {
        Pages::Home
    }
}

impl std::fmt::Display for Pages {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Pages::Home => write!(f, "Home"),
        }
    }
}

pub fn pages_header(top_state: &crate::App) -> Element<'_, PageMessage> {
    // let mut header: Vec<_> = Pages::options()
    //     .into_iter()
    //     .map(|page| {
    //         let is_selected = top_state.saved_state.page == page;
    //         let button = button(text(page.to_string()))
    //             .on_press(PageMessage::SetPage(page))
    //             .style(if is_selected {
    //                 iced::widget::button::primary
    //             } else {
    //                 iced::widget::button::secondary
    //             });
    //         button.into()
    //     })
    //     .collect();
    let mut header = vec![];

    header.push(text!("Version: {}-{}", RECAP_VERSION, GIT_COMMIT).into());

    header.push(
        container(text!(
            "Ram: {}/{}, Global Ram: {}/{}, Process CPU: {}%, Global CPU: {}%",
            format_bytes(top_state.system_info.ram_usage),
            format_bytes(top_state.system_info.ram_total),
            format_bytes(top_state.system_info.global_ram_usage),
            format_bytes(top_state.system_info.ram_total),
            (top_state.system_info.cpu_usage / top_state.system_info.number_of_cores as f32)
                .round(),
            top_state.system_info.global_cpu_usage.round(),
        ))
        .into(),
    );

    widget::Column::from_vec(header)
        .padding([0, 10])
        .spacing(10)
        .into()
}

pub fn update(state: &mut crate::App, message: PageMessage) -> Task<PageMessage> {
    match message {
        PageMessage::SetPage(page) => {
            state.saved_state.page = page;
        }
    };
    Task::none()
}

fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} Bytes")
    }
}
