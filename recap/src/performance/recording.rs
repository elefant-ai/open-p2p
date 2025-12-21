use iced::{
    Element, Length, Task,
    widget::{canvas, column, row, text},
};
use metrics::Label;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use snowline::prelude::*;
use tracing::{error, info};
use uuid::Uuid;

use crate::{
    handler::capture,
    metrics_impl::Snapshot,
    paths::get_paths,
    utils::action::{Action, ActionTask},
    widgets::system_info,
};

#[derive(Debug)]
struct Metric {
    basic_data: Vec<f64>,
    bar_data: [f64; 10],
    label: String,
    line_cache: iced::widget::canvas::Cache,
    bar_cache: iced::widget::canvas::Cache,
}

impl Metric {
    pub fn new(basic_data: Vec<f64>, label: String) -> Self {
        Self {
            bar_data: super::utils::calculate_histogram(&basic_data),
            basic_data: basic_data
                .into_par_iter()
                .map(|x| x * 1000.0)
                .collect::<Vec<_>>(),
            label,
            line_cache: iced::widget::canvas::Cache::new(),
            bar_cache: iced::widget::canvas::Cache::new(),
        }
    }

    pub fn view(&self) -> iced::Element<'_, Message> {
        let mut row = Vec::new();

        if let Some(line_graph) = self.line_graph() {
            row.push(line_graph);
        }

        if let Some(bar_graph) = self.bar_graph() {
            row.push(bar_graph);
        }

        if row.is_empty() {
            return iced::widget::column![].into();
        }

        column![
            text(&self.label).size(24),
            iced::widget::row(row).spacing(20)
        ]
        .into()
    }

    pub fn line_graph(&self) -> Option<iced::Element<'_, Message>> {
        if self.basic_data.is_empty() || self.basic_data.iter().all(|&x| x == 0.0) {
            return None; // No data to display
        }
        Some(
            Element::from(
                canvas(
                    LineGraph::new(self.basic_data.iter().cloned(), &self.line_cache)
                        .point_color_fn(|param| {
                            let value = param.value;
                            let theme = param.theme;
                            if value <= 0.05 {
                                // less then 50 ms
                                theme.extended_palette().success.base.color
                            } else if value > 0.05 && value <= 0.06 {
                                theme.extended_palette().warning.base.color
                            } else {
                                theme.extended_palette().danger.base.color
                            }
                        })
                        .unit_suffix("ms"),
                )
                .width(Length::Fixed(600.0)) // Increased for better visibility
                .height(Length::Fixed(300.0)), // Increased for better proportions
            )
            .map(Message::LineInteraction),
        )
    }

    pub fn bar_graph(&self) -> Option<iced::Element<'_, Message>> {
        if self.bar_data.is_empty() || self.bar_data.iter().all(|&x| x == 0.0) {
            return None; // No data to display
        }

        Some(
            Element::from(
                canvas(
                    BarGraph::new(self.bar_data.into_iter(), &self.bar_cache)
                        .bar_color_fn(|param| {
                            let value = param.index;
                            let theme = param.theme;
                            if value <= 5 {
                                // less then 50 ms
                                theme.extended_palette().success.base.color
                            } else if value <= 6 {
                                theme.extended_palette().warning.base.color
                            } else {
                                theme.extended_palette().danger.base.color
                            }
                        })
                        .bar_width(60.0) // Much wider bars for better visibility
                        .show_grid(true)
                        .show_labels(true)
                        .base_bars(10.0),
                )
                .width(Length::Fixed(600.0)) // Increased for better visibility
                .height(Length::Fixed(300.0)), // Increased for better proportions
            )
            .map(Message::BarInteraction),
        )
    }
}

#[derive(Debug)]
struct LineGraphOnly {
    basic_data: Vec<f64>,
    label: String,
    unit: String,
    line_cache: iced::widget::canvas::Cache,
}

impl LineGraphOnly {
    pub fn new(basic_data: Vec<f64>, label: String, unit: String) -> Self {
        Self {
            basic_data,
            label,
            unit,
            line_cache: iced::widget::canvas::Cache::new(),
        }
    }

    pub fn view(&self) -> iced::Element<'_, Message> {
        if self.basic_data.is_empty() || self.basic_data.iter().all(|&x| x == 0.0) {
            return column![].into(); // No data to display
        }

        Element::from(
            canvas(
                LineGraph::new(self.basic_data.iter().cloned(), &self.line_cache)
                    .single_point_color(iced::Color::from_rgb8(0, 255, 0))
                    .title_text(Some(self.label.clone()))
                    .unit_suffix(self.unit.clone()),
            )
            .width(Length::Fill) // Increased for better visibility
            .height(Length::Fixed(200.0)), // Increased for better proportions
        )
        .map(Message::LineInteraction)
    }
}

#[derive(Debug)]
pub struct RecordingPerformance {
    id: Uuid,
    inference_latency: Metric,
    inference_frame_interval: Metric,
    new_data_interval: Metric,
    cpu_usage: LineGraphOnly,
    total_cpu_usage: LineGraphOnly,
    ram_usage: LineGraphOnly,
    total_ram_usage: LineGraphOnly,
    encoding_latency: Metric,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RecordingStorage {
    encoding_latency: Vec<f64>,
    inference_latency: Vec<f64>,
    inference_frame_interval: Vec<f64>,
    new_data_interval: Vec<f64>,
    cpu_usage: Vec<f64>,
    total_cpu_usage: Vec<f64>,
    ram_usage: Vec<f64>,
    total_ram_usage: Vec<f64>,
}

impl RecordingStorage {
    pub fn is_empty(&self) -> bool {
        self.inference_latency.is_empty()
            && self.inference_frame_interval.is_empty()
            && self.new_data_interval.is_empty()
            && self.cpu_usage.is_empty()
            && self.total_cpu_usage.is_empty()
            && self.ram_usage.is_empty()
            && self.total_ram_usage.is_empty()
    }

    pub async fn load(id: Uuid) -> anyhow::Result<Self> {
        let file = get_paths()
            .recordings_dir
            .join(id.to_string())
            .join("metrics.json");
        let data = tokio::fs::read_to_string(file).await?;
        let storage = serde_json::from_str::<RecordingStorage>(&data)?;
        Ok(storage)
    }

    pub async fn save(&self, id: Uuid) -> anyhow::Result<()> {
        let file = get_paths()
            .recordings_dir
            .join(id.to_string())
            .join("metrics.json");
        let data = serde_json::to_string_pretty(self)?;
        tokio::fs::write(file, data).await?;
        Ok(())
    }

    pub fn get_data_from_snapshot(snapshot: &Snapshot, id: Uuid) -> RecordingStorage {
        let encoding_latency = snapshot
            .view_histogram(
                capture::ENCODING_LATENCY,
                &[Label::new("id", id.to_string())],
            )
            .map(Vec::from)
            .unwrap_or_default();
        let inference_latency = snapshot
            .view_histogram(
                capture::INFERENCE_LATENCY,
                &[Label::new("id", id.to_string())],
            )
            .map(Vec::from)
            .unwrap_or_default();
        let inference_frame_interval = snapshot
            .view_histogram(
                capture::INFERENCE_FRAME_INTERVAL,
                &[Label::new("id", id.to_string())],
            )
            .map(Vec::from)
            .unwrap_or_default();
        let new_data_interval = snapshot
            .view_histogram(
                capture::NEW_DATA_INTERVAL,
                &[Label::new("id", id.to_string())],
            )
            .map(Vec::from)
            .unwrap_or_default();

        let cpu_usage = snapshot
            .view_histogram(system_info::CPU_USAGE, &[Label::new("id", id.to_string())])
            .map(Vec::from)
            .unwrap_or_default();

        let total_cpu_usage = snapshot
            .view_histogram(
                system_info::TOTAL_CPU_USAGE,
                &[Label::new("id", id.to_string())],
            )
            .map(Vec::from)
            .unwrap_or_default();

        let ram_usage = snapshot
            .view_histogram(system_info::RAM_USAGE, &[Label::new("id", id.to_string())])
            .map(Vec::from)
            .unwrap_or_default();

        let total_ram_usage = snapshot
            .view_histogram(
                system_info::TOTAL_RAM_USAGE,
                &[Label::new("id", id.to_string())],
            )
            .map(Vec::from)
            .unwrap_or_default();

        RecordingStorage {
            encoding_latency,
            inference_latency,
            inference_frame_interval,
            new_data_interval,
            cpu_usage,
            total_cpu_usage,
            ram_usage,
            total_ram_usage,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Message {
    LineInteraction(snowline::line_graph::Interaction),
    BarInteraction(snowline::bar_graph::Interaction),
    SetData(RecordingStorage),
    GoHome,
    Empty,
}

impl RecordingPerformance {
    pub fn new(top_state: &crate::App, id: Uuid) -> (Self, Task<Message>) {
        let snapshot = RecordingStorage::get_data_from_snapshot(&top_state.snapshot, id);
        let task = if snapshot.is_empty() {
            Task::future(async move {
                match RecordingStorage::load(id).await {
                    Ok(loaded) => {
                        info!("Loaded recording metrics from disk");
                        Message::SetData(loaded)
                    }
                    Err(e) => {
                        error!("Failed to load recording metrics: {}", e);
                        Message::Empty
                    }
                }
            })
        } else {
            let storage = snapshot.clone();
            Task::future(async move {
                storage
                    .save(id)
                    .await
                    .unwrap_or_else(|e| error!("Failed to save metrics: {}", e));
                Message::Empty
            })
        };

        (
            Self {
                cpu_usage: LineGraphOnly::new(
                    snapshot.cpu_usage,
                    "CPU Usage (%)".into(),
                    "%".to_string(),
                ),
                total_cpu_usage: LineGraphOnly::new(
                    snapshot.total_cpu_usage,
                    "Total CPU Usage (%)".into(),
                    "%".to_string(),
                ),
                ram_usage: LineGraphOnly::new(
                    snapshot.ram_usage,
                    "RAM Usage (MiB)".into(),
                    "MiB".to_string(),
                ),
                total_ram_usage: LineGraphOnly::new(
                    snapshot.total_ram_usage,
                    "Total RAM Usage (MiB)".into(),
                    "MiB".to_string(),
                ),
                id,
                inference_latency: Metric::new(
                    snapshot.inference_latency,
                    "Inference Latency".into(),
                ),
                inference_frame_interval: Metric::new(
                    snapshot.inference_frame_interval,
                    "Inference Frame Interval".into(),
                ),
                new_data_interval: Metric::new(
                    snapshot.new_data_interval,
                    "New Data Interval".into(),
                ),
                encoding_latency: Metric::new(snapshot.encoding_latency, "Encoding Latency".into()),
            },
            task,
        )
    }

    pub fn update(&mut self, message: Message) -> ActionTask<Message> {
        match message {
            Message::SetData(data) => {
                self.inference_latency =
                    Metric::new(data.inference_latency, "Inference Latency".into());
                self.inference_frame_interval = Metric::new(
                    data.inference_frame_interval,
                    "Inference Frame Interval".into(),
                );
                self.new_data_interval =
                    Metric::new(data.new_data_interval, "New Data Interval".into());
                self.cpu_usage =
                    LineGraphOnly::new(data.cpu_usage, "CPU Usage (%)".into(), "%".to_string());
                self.total_cpu_usage = LineGraphOnly::new(
                    data.total_cpu_usage,
                    "Total CPU Usage (%)".into(),
                    "%".to_string(),
                );
                self.ram_usage =
                    LineGraphOnly::new(data.ram_usage, "RAM Usage (MiB)".into(), "MiB".to_string());
                self.total_ram_usage = LineGraphOnly::new(
                    data.total_ram_usage,
                    "Total RAM Usage (MiB)".into(),
                    "MiB".to_string(),
                );
            }
            Message::Empty => {
                // Handle save recording
            }
            Message::LineInteraction(_) => {
                // Handle line interaction
            }
            Message::BarInteraction(_) => {
                // Handle bar interaction
            }
            Message::GoHome => {
                return crate::Message::SetRecordingPerformance(None).tat();
            }
        }
        Task::none().tat()
    }

    pub fn view(&self) -> iced::Element<'_, Message> {
        let mut page: Vec<Element<'_, Message>> = Vec::new();
        page.push(
            iced::widget::button("go home")
                .on_press(Message::GoHome)
                .into(),
        );
        page.push(iced::widget::text(format!("Recording ID: {}", self.id)).into());

        page.push(
            column![
                text("System Usage"),
                row![
                    self.cpu_usage.view(),
                    self.total_cpu_usage.view(),
                    self.ram_usage.view(),
                    self.total_ram_usage.view(),
                ]
            ]
            .into(),
        );

        page.push(self.inference_latency.view());
        page.push(self.inference_frame_interval.view());
        page.push(self.new_data_interval.view());
        page.push(self.encoding_latency.view());

        iced::widget::scrollable(
            iced::widget::Column::with_children(page)
                .spacing(20)
                .padding(20),
        )
        .into()
    }
}
