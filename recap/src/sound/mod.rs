mod beep;
mod file_source;
pub use beep::{beep, double_beep, long_beep};
pub use file_source::FileSource;

use std::sync::LazyLock;

use rodio::{Source, cpal::FromSample};

static PLAYER: LazyLock<Player> = LazyLock::new(Player::new);

#[derive(derive_more::Debug)]
pub enum ControlMessages {
    #[debug("AppendSource")]
    AppendSource(Box<dyn rodio::Source<Item = f32> + Send + 'static>),
}

#[derive(Debug)]
struct Player {
    tx: std::sync::mpsc::Sender<ControlMessages>,
}

impl Player {
    pub fn new() -> Self {
        let (tx, rx) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            let stream_handle = rodio::OutputStreamBuilder::open_default_stream().unwrap();
            let rodio_sink = rodio::Sink::connect_new(stream_handle.mixer());

            rodio_sink.play();

            while let Ok(msg) = rx.recv() {
                match msg {
                    ControlMessages::AppendSource(source) => {
                        rodio_sink.append(source);
                    }
                }
            }
        });
        Self { tx }
    }
}

pub fn append_source<S>(source: S)
where
    S: Source + Send + 'static,
    f32: FromSample<S::Item>,
{
    PLAYER
        .tx
        .send(ControlMessages::AppendSource(Box::new(source)))
        .unwrap();
}
