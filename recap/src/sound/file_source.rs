// Store sound files as static byte arrays
mod sound_files {
    macro_rules! file_source {
        ($name:expr) => {
            include_bytes!(concat!("../../assets/", $name))
        };
    }

    pub static CAPTURE_FINISHED: &[u8] = file_source!("finished-capture.mp3");
    pub static CAPTURE_FAILED: &[u8] = file_source!("capture-error.mp3");
    pub static STARTING_CAPTURE: &[u8] = file_source!("starting-capture.mp3");
    pub static STARTING_INFERENCE: &[u8] = file_source!("starting-inference.mp3");
    pub static STOPPED_INFERENCE: &[u8] = file_source!("stopped-inference.mp3");
    pub static INFERENCE_FAILED: &[u8] = file_source!("inference-failed.mp3");
    pub static MODEL_CONTROL_STARTED: &[u8] = file_source!("model-control-started.mp3");
    pub static MODEL_CONTROL_STOPPED: &[u8] = file_source!("model-control-stopped.mp3");
    pub static STARTING_CAPTURE_WITH_INFERENCE: &[u8] =
        file_source!("starting-capture-with-inference.mp3");
    pub static INFERENCE_SLOW: &[u8] = file_source!("inference-slow.mp3");
    pub static COMMA_EQUAL_ON_START_ERROR: &[u8] = file_source!("error-comma-equal-on-start.mp3");
}

#[derive(Debug)]
pub enum FileSource {
    CaptureFinished,
    CaptureFailed,
    StartingCapture,
    StartingInference,
    StoppedInference,
    InferenceFailed,
    ModelControlStarted,
    ModelControlStopped,
    StartingCaptureWithInference,
    InferenceSlow,
    CommaEqualOnStartError,
}

impl FileSource {
    /// Play the sound
    pub fn play(self) {
        // Get the appropriate sound file
        let file = match self {
            FileSource::CaptureFinished => sound_files::CAPTURE_FINISHED,
            FileSource::CaptureFailed => sound_files::CAPTURE_FAILED,
            FileSource::StartingCapture => sound_files::STARTING_CAPTURE,
            FileSource::StartingInference => sound_files::STARTING_INFERENCE,
            FileSource::StoppedInference => sound_files::STOPPED_INFERENCE,
            FileSource::InferenceFailed => sound_files::INFERENCE_FAILED,
            FileSource::ModelControlStarted => sound_files::MODEL_CONTROL_STARTED,
            FileSource::ModelControlStopped => sound_files::MODEL_CONTROL_STOPPED,
            FileSource::StartingCaptureWithInference => {
                sound_files::STARTING_CAPTURE_WITH_INFERENCE
            }
            FileSource::InferenceSlow => sound_files::INFERENCE_SLOW,
            FileSource::CommaEqualOnStartError => sound_files::COMMA_EQUAL_ON_START_ERROR,
        };

        // Play the sound in a separate thread
        std::thread::spawn(move || {
            let source = rodio::Decoder::new(std::io::Cursor::new(file)).unwrap();
            super::append_source(source);
        });
    }
}
