use iced::futures::SinkExt as _;
use metrics::{Histogram, histogram};
use sysinfo::get_current_pid;
use tokio::time::sleep;
use uuid::Uuid;

pub const RAM_USAGE: &str = "ram_usage";
pub const TOTAL_RAM_USAGE: &str = "total_ram_usage";
pub const CPU_USAGE: &str = "cpu_usage";
pub const TOTAL_CPU_USAGE: &str = "total_cpu_usage";

#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub ram_total: u64,
    pub global_ram_usage: u64,
    pub ram_usage: u64,
    pub cpu_usage: f32,
    pub global_cpu_usage: f32,
    pub number_of_cores: u32,
    ram_usage_histogram: Histogram,
    global_ram_usage_histogram: Histogram,
    cpu_usage_histogram: Histogram,
    global_cpu_usage_histogram: Histogram,
}

#[derive(Debug, Clone)]
pub struct SystamInfoUpdate {
    pub global_ram_usage: u64,
    pub ram_usage: u64,
    pub cpu_usage: f32,
    pub global_cpu_usage: f32,
}

#[derive(Debug, Clone)]
pub enum SystemUpdate {
    Update(SystamInfoUpdate),
    SetId(Option<Uuid>),
}

impl Default for SystemInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemInfo {
    pub fn new() -> Self {
        let current_pid = get_current_pid().unwrap();
        let sys = sysinfo::System::new_all();
        let process = sys.process(current_pid).unwrap();
        let core_count = sys.cpus().len() as u32;
        Self {
            ram_total: sys.total_memory(),
            global_ram_usage: sys.used_memory(),
            ram_usage: process.memory(),
            cpu_usage: process.cpu_usage(),
            global_cpu_usage: sys.global_cpu_usage(),
            number_of_cores: core_count,
            ram_usage_histogram: histogram!(RAM_USAGE),
            global_ram_usage_histogram: histogram!(TOTAL_RAM_USAGE),
            cpu_usage_histogram: histogram!(CPU_USAGE),
            global_cpu_usage_histogram: histogram!(TOTAL_CPU_USAGE),
        }
    }
}

pub fn update(state: &mut SystemInfo, message: SystemUpdate) {
    match message {
        SystemUpdate::Update(info) => {
            state.global_ram_usage = info.global_ram_usage;
            state.ram_usage = info.ram_usage;
            state.cpu_usage = info.cpu_usage;
            state.global_cpu_usage = info.global_cpu_usage;
            // Record RAM usage in MiB for more readable graphs
            let to_mib = |bytes: u64| (bytes as f64) / (1024.0 * 1024.0);
            state.ram_usage_histogram.record(to_mib(state.ram_usage));
            state
                .global_ram_usage_histogram
                .record(to_mib(state.global_ram_usage));
            state
                .cpu_usage_histogram
                .record(state.cpu_usage / state.number_of_cores as f32);
            state
                .global_cpu_usage_histogram
                .record(state.global_cpu_usage);
        }
        SystemUpdate::SetId(id) => {
            if let Some(id) = id {
                state.ram_usage_histogram = histogram!(RAM_USAGE, "id" => id.to_string());
                state.global_ram_usage_histogram =
                    histogram!(TOTAL_RAM_USAGE, "id" => id.to_string());
                state.cpu_usage_histogram = histogram!(CPU_USAGE, "id" => id.to_string());
                state.global_cpu_usage_histogram =
                    histogram!(TOTAL_CPU_USAGE, "id" => id.to_string());
            } else {
                state.ram_usage_histogram = histogram!(RAM_USAGE);
                state.global_ram_usage_histogram = histogram!(TOTAL_RAM_USAGE);
                state.cpu_usage_histogram = histogram!(CPU_USAGE);
                state.global_cpu_usage_histogram = histogram!(TOTAL_CPU_USAGE);
            }
        }
    }
}

pub fn subscription() -> iced::Subscription<SystemUpdate> {
    iced::Subscription::run(|| {
        iced::stream::channel(
            2,
            |mut output: iced::futures::channel::mpsc::Sender<SystemUpdate>| async move {
                let mut sys = sysinfo::System::new_all();
                let current_pid = get_current_pid().unwrap();
                loop {
                    sleep(std::time::Duration::from_secs(1)).await;
                    sys.refresh_all();
                    let global_cpu_usage = sys.global_cpu_usage();
                    let process = sys.process(current_pid).unwrap();
                    let global_ram_usage = sys.used_memory();
                    let ram_usage = process.memory();
                    let cpu_usage = process.cpu_usage();
                    output
                        .send(SystemUpdate::Update(SystamInfoUpdate {
                            global_ram_usage,
                            ram_usage,
                            cpu_usage,
                            global_cpu_usage,
                        }))
                        .await
                        .unwrap();
                }
            },
        )
    })
}
