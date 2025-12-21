use std::{fs::File, path::PathBuf, sync::atomic::AtomicBool};

use parking_lot::{RwLock, RwLockReadGuard};
use tracing_subscriber::{
    Layer as _,
    fmt::{MakeWriter, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
};

/// Initialize the logging system.
/// including the `tracing` and `ffmpeg` logging systems.
pub fn init() {
    let file_layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_writer(Writer::new())
        .with_filter(tracing_subscriber::EnvFilter::new(
            "info,recap=trace,video=trace",
        ));

    tracing_subscriber::registry()
        .with(file_layer)
        .with(
            tracing_subscriber::fmt::layer()
                .with_line_number(true)
                .with_span_events(FmtSpan::CLOSE)
                .with_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or("warn,iced_wgpu::window::compositor=info,recap=debug".into()),
                ),
        )
        .init();
}

static RUNNING_WRITER: std::sync::LazyLock<(AtomicBool, RwLock<Option<File>>, AtomicBool)> =
    std::sync::LazyLock::new(|| {
        (
            AtomicBool::new(false),
            RwLock::new(None),
            AtomicBool::new(false),
        )
    });
struct Writer {
    main_file: std::fs::File,
    other_files: RwLock<Option<std::fs::File>>,
}

enum FileWriter<'a> {
    Main(&'a std::fs::File),
    Other(RwLockReadGuard<'a, Option<std::fs::File>>),
}

impl Writer {
    fn new() -> Self {
        let main_file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open("recap.log")
            .unwrap();
        Self {
            main_file,
            other_files: RwLock::default(),
        }
    }
}

impl std::io::Write for FileWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Self::Main(file) => file.write(buf),
            Self::Other(file) => {
                let mut file = file.as_ref().unwrap();
                file.write(buf)
            }
        }
    }

    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        match self {
            Self::Main(file) => file.write_all(buf),
            Self::Other(file) => {
                let mut file = file.as_ref().unwrap();
                file.write_all(buf)
            }
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Self::Main(file) => file.flush(),
            Self::Other(file) => {
                let mut file = file.as_ref().unwrap();
                file.flush()
            }
        }
    }
}

impl<'a> MakeWriter<'a> for Writer {
    type Writer = FileWriter<'a>;

    fn make_writer(&'a self) -> Self::Writer {
        FileWriter::Main(&self.main_file)
    }

    fn make_writer_for(&'a self, meta: &tracing::Metadata<'_>) -> Self::Writer {
        if meta
            .module_path()
            .filter(|x| x.contains("capture") || x.contains("video"))
            .is_some()
            && RUNNING_WRITER.0.load(std::sync::atomic::Ordering::SeqCst)
        {
            {
                let file = self.other_files.read();
                if file.is_some() {
                    return FileWriter::Other(file);
                }
            }

            if let Some(file) = RUNNING_WRITER.1.write().take() {
                *self.other_files.write() = Some(file);
                return FileWriter::Other(self.other_files.read());
            }
        }
        if RUNNING_WRITER.2.load(std::sync::atomic::Ordering::SeqCst)
            && self.other_files.read().is_some()
        {
            *self.other_files.write() = None;
        }
        FileWriter::Main(&self.main_file)
    }
}

/// Start the logging system to create new files.
pub fn start_log_file(path: PathBuf) {
    let file = File::options()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .unwrap();
    let mut writer = RUNNING_WRITER.1.write();
    *writer = Some(file);
    RUNNING_WRITER
        .0
        .store(true, std::sync::atomic::Ordering::SeqCst);
    RUNNING_WRITER
        .2
        .store(false, std::sync::atomic::Ordering::SeqCst);
}

/// Stop the logging system from creating new files.
pub fn halt_log_file() {
    RUNNING_WRITER
        .0
        .store(false, std::sync::atomic::Ordering::SeqCst);
    RUNNING_WRITER
        .2
        .store(true, std::sync::atomic::Ordering::SeqCst);
}
