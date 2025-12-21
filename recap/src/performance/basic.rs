use iced::{Element, Length, Task, widget::canvas};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use snowline::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Graph {
    Line,
    Bar,
}

#[derive(Debug)]
pub struct Performance {
    line_cache: iced::widget::canvas::Cache,
    bar_cache: iced::widget::canvas::Cache,
    data: Vec<f64>,
    histogram_data: [f64; 10],
    selected_graph: Graph,
}

#[derive(Debug, Clone)]
pub enum Message {
    NewData(Vec<f64>),
    SetGraph(Graph),
    LineInteraction(snowline::line_graph::Interaction),
    BarInteraction(snowline::bar_graph::Interaction),
}

// fn load_data_from_disk() -> Vec<f64> {
//     let data = std::fs::read_to_string("performance_data.json").ok();
//     data.and_then(|file| serde_json::from_str(&file).ok())
//         .unwrap_or_default()
// }

impl Default for Performance {
    fn default() -> Self {
        Self::new()
    }
}

impl Performance {
    pub fn new() -> Self {
        // let data = load_data_from_disk();
        // let histogram_data = super::utils::calculate_histogram(&data);
        Self {
            line_cache: iced::widget::canvas::Cache::new(),
            bar_cache: iced::widget::canvas::Cache::new(),
            data: Vec::new(),
            histogram_data: [0.0; 10],
            selected_graph: Graph::Line,
        }
    }

    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::NewData(new_data) => {
                self.data = new_data
                    .into_par_iter()
                    .map(|x| x * 1000.0)
                    .collect::<Vec<_>>();
                self.histogram_data = super::utils::calculate_histogram(&self.data);
                self.line_cache.clear();
                self.bar_cache.clear();
            }
            Message::SetGraph(graph) => {
                self.selected_graph = graph;
            }
            Message::LineInteraction(_) => {
                // Handle line graph interaction
            }
            Message::BarInteraction(_) => {
                // Handle bar graph interaction
            }
        }
        Task::none()
    }

    pub fn view(&self) -> Element<'_, Message> {
        if self.data.is_empty() {
            return Element::from(iced::widget::text(""));
        }

        let data_iter = self.data.iter().copied();

        let buttons = Element::from(
            iced::widget::row![
                iced::widget::button("LineGraph").on_press(Graph::Line),
                iced::widget::button("BarGraph").on_press(Graph::Bar)
            ]
            .spacing(10),
        )
        .map(Message::SetGraph);

        let graph = match self.selected_graph {
            Graph::Line => Element::from(
                canvas(
                    LineGraph::new(data_iter, &self.line_cache)
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
                .width(Length::Fixed(450.0)) // Increased for better visibility
                .height(Length::Fixed(300.0)), // Increased for better proportions
            )
            .map(Message::LineInteraction),
            Graph::Bar => {
                Element::from(
                    canvas(
                        BarGraph::new(self.histogram_data.into_iter(), &self.bar_cache)
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
                    .width(Length::Fixed(450.0)) // Increased for better visibility
                    .height(Length::Fixed(300.0)), // Increased for better proportions
                )
                .map(Message::BarInteraction)
            }
        };

        let content = iced::widget::column![buttons, graph].spacing(10);
        content.into()
    }
}
