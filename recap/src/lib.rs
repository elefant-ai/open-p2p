mod external;
mod handler;
mod hot_key;
pub mod input_manager;
mod logger;
pub mod metrics_impl;
mod pages;
pub mod paths;
pub mod performance;
mod saved_state;
#[cfg(feature = "server")]
mod server;
pub mod snap_shot_state;
pub mod sound;
mod upload;
pub mod utils;
pub mod widgets;

use std::path::PathBuf;

use hashbrown::HashMap;
use iced::{
    Length, Task,
    widget::{self, container, text},
};
use metrics::Label;
use metrics_impl::{ExternalHandle, Snapshot};
use paths::get_paths;
use recap_gst::srcs::InputSrc;
use saved_state::SavedState;
use utils::windows::InnerWindow;
use uuid::Uuid;
use widgets::{meta_data::set_meta_data, set_mic_target, set_target};
use window_handling::WindowInfo;

pub fn run() -> anyhow::Result<()> {
    logger::init();

    input_manager::setup();

    recap_gst::gst::init().unwrap();

    let _ = handler::init_mouse()?;

    iced::application(App::new, App::update, App::view)
        .theme(App::theme)
        .title(App::title)
        .subscription(App::subscriptions)
        .font(iced_fonts::BOOTSTRAP_FONT_BYTES)
        .run_with_device_events(input_manager::handle_device_event)?;

    Ok(())
}

#[derive(Debug, Clone)]
pub enum Message {
    Refresh,
    SetTarget(InnerWindow),
    SetMicVolume(f64),
    SetMic(recap_gst::list_devices::Device),
    Performance(performance::basic::Message),
    SetTask(String),
    SetEnv(String),
    SaveSettings,
    SetEnvSubtype(String),
    SetUser(String),
    Uploader(upload::Message),
    CloseRequested,
    Exit,
    Page(pages::PageMessage),
    HotKey(hot_key::HotKey),
    SystemInfo(widgets::system_info::SystemUpdate),
    SetError(uuid::Uuid, Option<String>),
    Handler(handler::Message),
    CaptureFinished(uuid::Uuid),
    WindowSize(widgets::window_size::WindowSizeMessage),
    RunBack(PathBuf),
    QueryState(iced::futures::channel::mpsc::Sender<snap_shot_state::StateSnapshot>),
    SaveError(uuid::Uuid),
    RecordingPerformance(performance::recording::Message),
    SetRecordingPerformance(Option<Uuid>),
    #[allow(dead_code)]
    UpdateKeys,
    SetEnableMicAudio(bool),
}

#[derive(derive_more::Debug)]
pub struct App {
    pub devices: Vec<InnerWindow>,
    pub target: Option<InnerWindow>,
    pub mic: Option<recap_gst::list_devices::Device>,
    pub mic_devices: Vec<recap_gst::list_devices::Device>,
    pub error: Option<String>,
    pub uploader: upload::State,
    pub current_uuid: Option<uuid::Uuid>,
    pub saved_state: SavedState,
    pub system_info: widgets::system_info::SystemInfo,
    #[debug(skip)]
    pub clipboard: arboard::Clipboard,
    pub errors: HashMap<uuid::Uuid, Vec<String>>,
    pub handler: handler::State,
    pub cached_window_info: widgets::window_size::CachedWindowInfo,
    pub error_temp: HashMap<uuid::Uuid, u64>,
    pub keys: utils::view_state::KeyView,
    pub snapshot: Snapshot,
    pub metrics_handle: ExternalHandle,
    pub inference_latency: performance::basic::Performance,
    recording_performance: Option<performance::recording::RecordingPerformance>,
}

impl App {
    pub fn new() -> (Self, Task<Message>) {
        let metrics_handle = metrics_impl::init_metrics();

        let saved_state = std::fs::read_to_string(&get_paths().state_file)
            .ok()
            .and_then(|file| serde_json::from_str::<SavedState>(&file).ok())
            .unwrap_or_default();

        let uploader = upload::State::new();

        let (handler_state, handler_task) = handler::State::new();

        let input_options = recap_gst::srcs::DefaultSrc::get_input_options()
            .into_iter()
            .filter_map(|x| x.title().map(|title| InnerWindow::new(title, x)))
            .collect::<Vec<_>>();

        let mut errors = HashMap::new();

        for dir in get_paths().recordings_dir.read_dir().unwrap() {
            let dir = dir.unwrap();
            let uuid = Uuid::parse_str(&dir.file_name().to_string_lossy()).ok();
            if let Some(uuid) = uuid {
                let error_file = dir.path().join(upload::ERROR_STATE_FILENAME);
                if error_file.exists() {
                    if let Ok(error) = std::fs::read_to_string(&error_file) {
                        errors.insert(uuid, error.lines().map(String::from).collect());
                    }
                }
            }
        }

        let mic_devices =
            recap_gst::mic_to_mp3::Recorder::list_microphone_devices().unwrap_or_default();

        let state = App {
            target: input_options.iter().find_map(|window| {
                if Some(&window.title) == saved_state.target.as_ref() {
                    Some(window.clone())
                } else {
                    None
                }
            }),
            mic: mic_devices
                .iter()
                .find_map(|input| {
                    if Some(format!("{}:{}", input.name(), input.adaptor_name())) == saved_state.mic
                    {
                        Some(input.clone())
                    } else {
                        None
                    }
                })
                .or(mic_devices.first().cloned()),
            mic_devices,
            devices: input_options,
            error: None,
            uploader,
            current_uuid: None,
            saved_state,
            clipboard: arboard::Clipboard::new().unwrap(),
            system_info: widgets::system_info::SystemInfo::new(),
            errors,
            handler: handler_state,
            cached_window_info: widgets::window_size::CachedWindowInfo::default(),
            error_temp: HashMap::new(),
            keys: utils::view_state::KeyView::new(),
            snapshot: metrics_handle.snapshot(),
            metrics_handle,
            inference_latency: performance::basic::Performance::new(),
            recording_performance: None,
        };

        let tasks = Task::batch([handler_task.map(Message::Handler)]);

        (state, tasks)
    }

    pub fn theme(&self) -> iced::Theme {
        iced::Theme::Dracula
    }

    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::SetMicVolume(volume) => {
                self.saved_state.mic_volume = volume;
            }
            Message::SetEnableMicAudio(enabled) => {
                self.saved_state.enable_mic_audio = enabled;
            }
            Message::RecordingPerformance(msg) => {
                if let Some(recording_performance) = &mut self.recording_performance {
                    return recording_performance
                        .update(msg)
                        .handle(self, Message::RecordingPerformance);
                }
            }
            Message::SetRecordingPerformance(uuid) => {
                let Some(uuid) = uuid else {
                    self.recording_performance = None;
                    return Task::none();
                };
                let (pef, task) = performance::recording::RecordingPerformance::new(self, uuid);
                self.recording_performance = Some(pef);
                return task.map(Message::RecordingPerformance);
            }
            Message::UpdateKeys => {
                self.keys.update();
                let new_snap = self.metrics_handle.snapshot();
                self.snapshot.merge(new_snap);

                if let Some(current_id) = self.current_uuid
                    && let Some(latency) = self.snapshot.view_histogram(
                        handler::capture::INFERENCE_LATENCY,
                        &[Label::new("id", current_id.to_string())],
                    )
                {
                    return self
                        .inference_latency
                        .update(performance::basic::Message::NewData(latency.to_vec()))
                        .map(Message::Performance);
                }
            }
            Message::Performance(msg) => {
                return self.inference_latency.update(msg).map(Message::Performance);
            }
            Message::CaptureFinished(uuid) => {
                if let Some(current_uuid) = self.current_uuid {
                    if current_uuid == uuid {
                        self.current_uuid = None;
                    }
                }
            }
            Message::Handler(message) => {
                return handler::update(self, message);
            }
            Message::SetError(id, error) => {
                if let Some(error) = error {
                    self.errors.entry(id).or_default().push(error);
                    let current_temp = self.error_temp.entry(id).or_insert(0);
                    *current_temp += 1;
                    self.handler.reset();
                    return Task::perform(
                        tokio::time::sleep(std::time::Duration::from_secs(1)),
                        move |_| Message::SaveError(id),
                    );
                } else {
                    self.errors.remove(&id);
                }
            }
            Message::SaveError(id) => {
                let current_temp = self.error_temp.entry(id).or_insert(0);
                *current_temp = current_temp.saturating_sub(1);

                if *current_temp == 0 {
                    let errors = self.errors.get(&id).cloned().unwrap_or_default();
                    return Task::future(async move {
                        upload::save_error_state_to_disk(id, errors.join("\n")).await;
                    })
                    .discard();
                }
            }
            Message::SystemInfo(info) => {
                widgets::system_info::update(&mut self.system_info, info);
            }
            Message::HotKey(hotkeys) => return hot_key::update(self, hotkeys),
            Message::Page(message) => {
                return pages::update(self, message).map(Message::Page);
            }
            Message::CloseRequested => {
                if self.current_uuid.is_some() {
                    self.error = Some("Cannot close while recording".to_string());
                    return Task::none();
                }
                self.on_exit();
                return iced::exit();
            }
            Message::Exit => {
                self.on_exit();
                return iced::exit();
            }
            Message::Uploader(message) => {
                return upload::update(self, message).handle(self, Message::Uploader);
            }
            Message::Refresh => {
                self.devices = recap_gst::srcs::DefaultSrc::get_input_options()
                    .into_iter()
                    .filter_map(|x| x.title().map(|title| InnerWindow::new(title, x)))
                    .collect::<Vec<_>>();
                self.mic_devices =
                    recap_gst::mic_to_mp3::Recorder::list_microphone_devices().unwrap_or_default();
            }
            Message::SetTarget(target) => {
                self.saved_state.target = Some(target.title.clone());
                self.target = Some(target.clone());
                // Eagerly refresh window info when target changes
                self.cached_window_info.size = target.window.size().ok();
                self.cached_window_info.position = target.window.position().ok();
                self.cached_window_info.scale_factor = Some(target.window.scale_factor());
            }
            Message::SetMic(mic) => {
                self.saved_state.mic = Some(format!("{}:{}", mic.name(), mic.adaptor_name()));
                self.mic = Some(mic);
            }
            Message::SetUser(user) => {
                self.saved_state.user = user;
            }
            Message::SetTask(task) => {
                self.saved_state.task = task;
            }
            Message::SetEnv(env) => {
                self.saved_state.env = env;
            }
            Message::SetEnvSubtype(env_subtype) => {
                self.saved_state.env_subtype = env_subtype;
            }
            Message::SaveSettings => {
                self.saved_state.target = self
                    .target
                    .as_ref()
                    .and_then(|x| x.window.title())
                    .or(self.saved_state.target.clone());
                std::fs::write(
                    &get_paths().state_file,
                    serde_json::to_string(&self.saved_state).unwrap(),
                )
                .unwrap();
            }
            Message::WindowSize(message) => {
                return widgets::window_size::update_window_size(self, message);
            }
            Message::QueryState(mut sender) => match sender.try_send(self.into()) {
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!("Failed to send state snapshot: {:?}", e);
                }
            },
            Message::RunBack(path) => {
                return upload::update(self, upload::Message::RunBack(path))
                    .handle(self, Message::Uploader);
            }
        };

        Task::none()
    }

    pub fn view(&self) -> iced::Element<'_, Message> {
        if let Some(ref recording_performance) = self.recording_performance {
            return recording_performance
                .view()
                .map(Message::RecordingPerformance);
        }

        let target_select = set_target(self);
        let mic_select = set_mic_target(self);

        let meta_data = set_meta_data(self);

        let left_column = widget::column![
            if let Some(ref error) = self.error {
                text(format!("Error: {error}"))
            } else {
                widget::text("")
            },
            meta_data,
            widget::Space::with_height(30.0),
            widget::row![
                widget::column![text("Select a target:"), target_select],
                widget::column![widget::column![text("Select a microphone:"), mic_select]],
                widget::column![
                    widget::checkbox("Enable Mic Audio", self.saved_state.enable_mic_audio)
                        .on_toggle(Message::SetEnableMicAudio),
                    widget::slider(
                        0.0..=10.0,
                        self.saved_state.mic_volume,
                        Message::SetMicVolume
                    )
                    .step(0.2)
                    .width(150.0)
                ]
            ],
            widget::button("Refresh").on_press(Message::Refresh),
            widget::Space::with_height(30.0),
            upload::view(self).map(Message::Uploader),
        ];

        let mut home_page = widget::Row::new()
            .push(left_column)
            .padding(10.0)
            .spacing(20.0);

        if cfg!(feature = "trace") {
            home_page = home_page.push(widget::column![
                self.keys.view(),
                self.inference_latency.view().map(Message::Performance),
            ]);
        }

        if cfg!(feature = "window_info") {
            home_page = home_page.push(widget::column![
                widget::Space::with_height(10.0),
                widgets::window_size::window_size_display(self),
                widget::Space::with_height(10.0),
                widgets::window_size::window_size_control(self),
            ]);
        }

        let selected_page: iced::Element<'_, Message> = match self.saved_state.page {
            pages::Pages::Home => home_page.into(),
        };

        container(widget::column![
            pages::pages_header(self).map(Message::Page),
            container(selected_page).padding(5.0),
        ])
        .style(|theme| {
            iced::widget::container::Style::default().background(theme.palette().background)
        })
        .width(Length::Fill)
        .height(Length::Fill)
        .padding(20.0)
        .into()
    }

    fn on_exit(&mut self) {
        std::fs::write(
            &get_paths().state_file,
            serde_json::to_string(&self.saved_state).unwrap(),
        )
        .unwrap();
    }

    fn title(&self) -> String {
        format!("Recap - {}", self.saved_state.task)
    }

    fn subscriptions(&self) -> iced::Subscription<Message> {
        iced::Subscription::batch(vec![
            #[cfg(feature = "trace")]
            iced::time::every(std::time::Duration::from_secs(1)).map(|_| Message::UpdateKeys),
            hot_key::subscription(self),
            upload::subscription(&self.uploader).map(Message::Uploader),
            iced::window::close_requests().map(|_| Message::CloseRequested),
            widgets::system_info::subscription().map(Message::SystemInfo),
            external::subscription(self),
            #[cfg(feature = "server")]
            crate::server::subscription(),
            iced::Subscription::run(|| {
                iced::stream::channel(
                    1,
                    |mut output: iced::futures::channel::mpsc::Sender<Message>| async move {
                        ctrlc::set_handler(move || {
                            output.try_send(Message::Exit).unwrap();
                        })
                        .unwrap();
                    },
                )
            }),
        ])
    }
}
