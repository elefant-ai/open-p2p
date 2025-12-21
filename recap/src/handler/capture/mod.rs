mod helpers;
mod input;
mod lag_channel;
mod on_finish_check;

use std::collections::HashMap;
use std::time::Instant;

use anyhow::Context;
use helpers::{process_keys, process_mouse};
use iced::futures::future;
use iced::futures::future::Either;
use iced::futures::pin_mut;
pub use input::{InputFrame, InputFrameMouse, save_input_state};
use lag_channel::{Recv, lag_channel};
use metrics::Histogram;
use metrics::histogram;
use recap_gst::gst;
use recap_gst::gst::prelude::{ElementExt as _, PadExt as _};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32},
    },
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{debug, error, info, span, trace, warn};
use video_annotation_proto::video_annotation::VideoAnnotationMetadata;
use video_inference_grpc::video_inference::Action;
use video_inference_grpc::video_inference::Frame;
use window_handling::WindowInfo;

use fast_image_resize::CpuExtensions;
use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::pixels::PixelType;
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};

use crate::input_manager::set_inference_running;
use crate::{
    external::{send_error, send_message},
    input_manager::lift_simulated_keys,
    logger::{halt_log_file, start_log_file},
    sound::{FileSource, beep, double_beep},
    widgets::meta_data::{GIT_COMMIT, RECAP_VERSION},
};
use video_inference_grpc::prost::Message;

use windows::Win32::UI::WindowsAndMessaging::{
    SPI_GETMOUSE, SPI_SETMOUSE, SPIF_SENDCHANGE, SPIF_UPDATEINIFILE, SystemParametersInfoW,
};

static MODEL_INPUT_HEIGHT: u32 = 192;
static MODEL_INPUT_WIDTH: u32 = 192;

pub const INFERENCE_LATENCY: &str = "inference_latency";
pub const NEW_DATA_INTERVAL: &str = "new_data_interval";
pub const INFERENCE_FRAME_INTERVAL: &str = "inference_frame_interval";
pub const ENCODING_LATENCY: &str = "encoding_latency";

pub fn get_mouse_acceleration() -> windows::core::Result<bool> {
    #[allow(unsafe_code)]
    unsafe {
        // 1) Read current settings
        let mut params = [0i32; 3];
        SystemParametersInfoW(
            SPI_GETMOUSE,
            0,
            Some(params.as_mut_ptr() as *mut _),
            windows::Win32::UI::WindowsAndMessaging::SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS(0),
        )?;

        // 2) Check if acceleration is enabled (third parameter != 0)
        Ok(params[2] != 0)
    }
}

pub fn set_mouse_acceleration(enabled: bool) -> windows::core::Result<()> {
    #[allow(unsafe_code)]
    unsafe {
        // 1) Read current settings
        let mut params = [0i32; 3];
        SystemParametersInfoW(
            SPI_GETMOUSE,
            0,
            Some(params.as_mut_ptr() as *mut _),
            windows::Win32::UI::WindowsAndMessaging::SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS(0),
        )?;

        let enabled = enabled as i32;

        if enabled != params[2] {
            params[2] = enabled; // Set acceleration flag
            SystemParametersInfoW(
                SPI_SETMOUSE,
                0,
                Some(params.as_mut_ptr() as *mut _),
                SPIF_UPDATEINIFILE | SPIF_SENDCHANGE,
            )?;
        }
    }
    Ok(())
}

#[derive(Debug)]
pub struct Capture {
    running: Arc<AtomicBool>,
    inference_enabled: Arc<AtomicBool>,
    started_inference: bool,
    stop_capture_notify: Arc<tokio::sync::Notify>,
}

impl Capture {
    pub fn new() -> Result<Self, anyhow::Error> {
        Ok(Capture {
            inference_enabled: Arc::new(AtomicBool::new(false)),
            started_inference: false,
            running: Arc::new(AtomicBool::new(false)),
            stop_capture_notify: Arc::new(tokio::sync::Notify::new()),
        })
    }

    pub fn reset(&mut self) {
        self.stop_capture().expect("Failed to stop capture");
        self.running
            .store(false, std::sync::atomic::Ordering::SeqCst);
        self.inference_enabled
            .store(false, std::sync::atomic::Ordering::SeqCst);
        self.started_inference = false;
        self.stop_capture_notify = Arc::new(tokio::sync::Notify::new());
    }

    pub fn start_capture<W: WindowInfo + 'static>(
        &mut self,
        id: uuid::Uuid,
        target: W,
        path: impl AsRef<Path>,
        meta_data: VideoAnnotationMetadata,
        with_inference: bool,
        device: Option<recap_gst::list_devices::Device>,
        mic_volume: f64,
    ) -> Result<(), anyhow::Error> {
        if self.running.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(anyhow::anyhow!("Capture already running"));
        } else {
            self.running
                .store(true, std::sync::atomic::Ordering::SeqCst);
            self.started_inference = false;
            self.stop_capture_notify = Arc::new(tokio::sync::Notify::new());
            // Reset latency stats when starting a new capture
        }

        // Check mouse capture is disabled.
        // TODO: enable/disable mouse capture on each start.
        match get_mouse_acceleration() {
            Ok(enabled) => {
                if enabled {
                    return Err(anyhow::anyhow!(
                        "Mouse acceleration is somehow enabled. It should have been disabled on app start. Please restart the app."
                    ));
                }
            }
            Err(e) => {
                error!("Failed to get mouse acceleration: {:?}", e);
                return Err(anyhow::anyhow!("Failed to get mouse acceleration: {e:?}"));
            }
        };

        let path = path.as_ref().to_owned();
        let running = self.running.clone();
        let stop_capture_notify = self.stop_capture_notify.clone();

        let handle = tokio::runtime::Handle::current();
        // // enable to run the capture for 60 seconds and then stop it. used for checking file size
        // std::thread::spawn({
        //     let stop_notify = stop_capture_notify.clone();
        //     move || {
        //         std::thread::sleep(std::time::Duration::from_secs(60));
        //         stop_notify.notify_one();
        //     }
        // });

        let with_inference = if with_inference {
            self.inference_enabled
                .store(true, std::sync::atomic::Ordering::Relaxed);

            Some(self.inference_enabled.clone())
        } else {
            self.inference_enabled
                .store(false, std::sync::atomic::Ordering::Relaxed);
            None
        };

        std::thread::spawn(move || {
            beep();
            if with_inference.is_some() {
                FileSource::StartingCaptureWithInference.play();
            } else {
                FileSource::StartingCapture.play();
            }
            start_log_file(path.join("capture.log"));
            let _guard = span!(tracing::Level::INFO, "capture thread").entered();
            if let Err(err) = start_capture(
                id,
                target,
                path,
                meta_data,
                handle,
                stop_capture_notify,
                with_inference,
                device,
                mic_volume,
            ) {
                error!("Error in capture thread: {:?}", err);
                send_error(id, Some(format!("{err:#}")));
                FileSource::CaptureFailed.play();
            } else {
                FileSource::CaptureFinished.play();
            }
            halt_log_file();
            // make sure stop is to true
            running.store(false, std::sync::atomic::Ordering::Relaxed);
            send_message(crate::Message::CaptureFinished(id));
        });

        Ok(())
    }

    pub fn toggle_model_control(&mut self) {
        let _ = self.inference_enabled.swap(
            !self
                .inference_enabled
                .load(std::sync::atomic::Ordering::SeqCst),
            std::sync::atomic::Ordering::SeqCst,
        );
    }

    // This is called from the main thread.
    pub fn stop_inference(&mut self) {
        self.inference_enabled
            .store(false, std::sync::atomic::Ordering::Relaxed);

        info!("Final inference latency statistics:");
        double_beep();
    }

    pub fn stop_capture(&mut self) -> Result<(), anyhow::Error> {
        if self
            .inference_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            // If inference is enabled, notify the inference task to stop
            self.stop_inference();
        }
        self.stop_capture_notify.notify_waiters();
        Ok(())
    }
}

async fn send_inference_frames(
    recv: Recv<Frame>,
    mut writer: tokio::io::WriteHalf<wsl_tools::SocatStream>,
    mut shutdown_rx: tokio::sync::oneshot::Receiver<()>,
    timer: Arc<Mutex<HashMap<i32, Instant>>>,
) -> Result<(), anyhow::Error> {
    loop {
        // Use tokio::select to check both the frame channel and the shutdown signal
        tokio::select! {
            frame_result = recv.recv() => {
                match frame_result {
                    Ok(frame) => {
                        let encoded = frame.encode_to_vec();
                        let len = encoded.len() as u32;
                        info!("Sending frame with id {} and length {}", frame.id, len);
                        timer
                            .lock()
                            .unwrap()
                            .insert(frame.id, std::time::Instant::now());
                        writer
                            .write_all(&len.to_le_bytes())
                            .await
                            .context("failed to write length")?;
                        writer
                            .write_all(&encoded)
                            .await
                            .context("failed to write frame")?;
                    },
                    Err(_) => {
                        debug!("Frame channel closed, exiting writer task");
                        break;
                    }
                }
            },
            _ = &mut shutdown_rx => {
                debug!("Shutdown signal received, exiting frame sending task");
                break;
            }
        }
    }
    anyhow::Ok(())
}

async fn receive_inference_actions(
    reader: &mut tokio::io::ReadHalf<wsl_tools::SocatStream>,
    timer: &Arc<Mutex<HashMap<i32, Instant>>>,
    latency: &Histogram,
) -> Result<Action, anyhow::Error> {
    let mut length_buffer = [0u8; 4];
    match reader
        .read_exact(&mut length_buffer)
        .await
        .context("failed to read length")
        .map_err(|e| {
            error!("Error reading length: {:?}", e);
        }) {
        Ok(_) => {}
        Err(_) => {
            error!("Failed to read length, returning empty action");
            return Err(anyhow::anyhow!("Failed to read length"));
        }
    };
    let length = u32::from_le_bytes(length_buffer) as usize;
    let mut action_buffer = vec![0u8; length];
    match reader
        .read_exact(&mut action_buffer)
        .await
        .context("failed to read action")
    {
        Ok(_) => {}
        Err(e) => {
            error!("Error reading action: {:?}", e);
            return Err(anyhow::anyhow!("Failed to read action"));
        }
    };
    let finished_reading_now = std::time::Instant::now();
    let action =
        match video_inference_grpc::video_inference::Action::decode(action_buffer.as_slice()) {
            Ok(action) => action,
            Err(e) => {
                error!("Failed to decode action: {:?}", e);
                return Err(anyhow::anyhow!("Failed to decode action"));
            }
        };
    let action_id = action.id;

    if let Some(start) = timer.lock().unwrap().remove(&action_id) {
        let elapsed: std::time::Duration = finished_reading_now.duration_since(start);
        let latency_ms = elapsed.as_secs_f64();
        latency.record(latency_ms);
        trace!("Received action after {:?} from server", elapsed);
    } else {
        warn!(
            "Received action with id {} but no start time found",
            action_id
        );
    }
    Ok(action)
}

#[allow(clippy::too_many_arguments)]
fn start_capture<W: WindowInfo>(
    id: uuid::Uuid,
    target: W,
    path: PathBuf,
    meta_data: VideoAnnotationMetadata,
    handle: tokio::runtime::Handle,
    stop_capture_notify: Arc<tokio::sync::Notify>,
    with_inference: Option<Arc<AtomicBool>>,
    audio_device: Option<recap_gst::list_devices::Device>,
    mic_volume: f64,
) -> Result<(), anyhow::Error> {
    // Signal to stop the inference stream
    let stop_inference_signal = Arc::new(tokio::sync::Notify::new());

    // special channel that has a buffer of 1 that will drop the oldest message if the buffer is full
    let (inference_sender, inference_recv) =
        lag_channel::<video_inference_grpc::video_inference::Frame>();

    let frame_timer = Arc::new(Mutex::new(HashMap::new()));
    let frame_timer_for_recv: Arc<Mutex<HashMap<i32, Instant>>> = frame_timer.clone();
    let latency = histogram!(INFERENCE_LATENCY, "id" => id.to_string());
    let stop_capture_notify = stop_capture_notify.clone();
    let stop_inference = stop_inference_signal.clone();

    let audio_key = handle.spawn({
        let stop_capture_notify = stop_capture_notify.clone();
        async move {
            let (sender, mut recv) =
                tokio::sync::mpsc::channel::<(bool, std::time::SystemTime)>(30);
            let mut currently_pressed = false;
            let id = crate::input_manager::listen(move |event, _| {
                if let crate::input_manager::Event::KeyboardInput {
                    pressed,
                    key: input_codes::Keycode::Quote,
                } = event.event
                {
                    if pressed != currently_pressed {
                        currently_pressed = pressed;
                        let _ = sender.try_send((pressed, event.time));
                    }
                }
            });
            let mut times: Vec<(bool, std::time::SystemTime)> = Vec::new();

            loop {
                let future1 = stop_capture_notify.notified();
                let future2 = recv.recv();
                pin_mut!(future1);
                pin_mut!(future2);

                match future::select(future1, future2).await {
                    Either::Right((Some(v), _)) => {
                        println!("{:?}", v);
                        times.push(v);
                    }
                    _ => break,
                };
            }

            crate::input_manager::remove_listener(id);

            debug!("Captured {} audio input events", times.len());

            times
        }
    });

    if let Some(with_inference) = with_inference.clone() {
        handle.spawn({
            let model_control_enabled = with_inference;
            let stop_capture_notify_for_inference = stop_capture_notify.clone();
            async move {
                let fut = async {
                    debug!("Starting inference rpc");

                    let stream = wsl_tools::SocatStream::connect("/tmp/uds.recap")?;
                    let (mut reader, writer) = stream.split();

                    let (writer_shutdown_tx, writer_shutdown_rx) = tokio::sync::oneshot::channel();

                    let writer_handle = tokio::spawn(async move {
                        let result = send_inference_frames(
                            inference_recv,
                            writer,
                            writer_shutdown_rx,
                            frame_timer.clone(),
                        )
                        .await;

                        if let Err(e) = result {
                            error!("Error in frame sender: {:?}", e);
                        }
                        debug!("Inference frame sender closed");
                    });

                    debug!("Started inference rpc");

                    let mut keys_pressed: Vec<String> = Vec::new();
                    let mut mouse_buttons_pressed: Vec<String> = Vec::new();

                    let model_action_fut = async {
                        loop {
                            let Action {
                                keys,
                                id,
                                mouse_action,
                            } = match receive_inference_actions(
                                &mut reader,
                                &frame_timer_for_recv,
                                &latency,
                            )
                            .await
                            {
                                Ok(action) => action,
                                Err(e) => {
                                    error!("Error receiving action: {:?}", e);
                                    return Err(e);
                                }
                            };

                            let model_control_enabled =
                                model_control_enabled.load(std::sync::atomic::Ordering::Relaxed);
                            trace!(
                                "action id {}. Keys {:?}. model_control_enabled:{}.",
                                id, keys, model_control_enabled
                            );
                            if model_control_enabled {
                                process_keys(keys, &mut keys_pressed);
                                if let Some(mouse_action) = mouse_action {
                                    process_mouse(mouse_action, &mut mouse_buttons_pressed);
                                }
                            } else {
                                trace!("Model control is disabled, clearing keys");
                                keys_pressed.clear();
                                mouse_buttons_pressed.clear();
                            }
                        }
                        #[expect(unreachable_code)]
                        anyhow::Ok(())
                    };

                    let result = tokio::select! {
                        res = model_action_fut => {
                            debug!("Inference stream closed");
                            res
                        }
                        _ = stop_inference_signal.notified() => {
                            debug!("Stopping inference stream");
                            // inference_enabled.store(false, std::sync::atomic::Ordering::SeqCst);
                            Ok(())
                        }
                    };
                    // send the shutdown signal to the writer
                    if let Err(e) = writer_shutdown_tx.send(()) {
                        error!("Failed to send shutdown signal to writer: {:?}", e);
                    }

                    match tokio::time::timeout(std::time::Duration::from_secs(2), writer_handle)
                        .await
                    {
                        Ok(_) => debug!("Writer task shut down successfully"),
                        Err(_) => warn!("Writer task shutdown timed out after 2 seconds"),
                    }

                    info!("inference rpc receiver stopped");
                    result
                };

                if let Err(e) = fut.await {
                    error!("Error in inference rpc receiver: {:?}", e);
                    send_error(id, Some(format!("{e:#}")));
                    FileSource::InferenceFailed.play();
                    // Notify to stop the capture when inference fails
                    info!("Stopping capture due to inference failure");
                    stop_capture_notify_for_inference.notify_one();
                }
            }
        });
    }

    if let Some(model_control_enabled) = with_inference.clone() {
        set_inference_running(Some(model_control_enabled));
    }

    // setup the scaler for the inference

    let video_path = path.join("video.mp4");

    let frame_count = Arc::new(AtomicU32::new(0));

    let input_state = Arc::new(Mutex::new(Vec::new()));

    let data_input_state = Arc::clone(&input_state);

    let fps = 20.0;

    const HEIGHT: i32 = 480;

    // Calculate the width based on the target size and the desired height with a 1/1  ratio.
    // We want the width to be a multiple of 4 for video encoding
    let calc_width = {
        let (size_width, size_height) = target.size().context("Failed to get target size")?;
        let scale_factor = HEIGHT as f64 / size_height as f64;
        let width = (size_width as f64 * scale_factor).round();
        next_multiple_of(width as i32, 4)
    };

    let pipeline_builder = recap_gst::record_window::PipelineBuilder::new()
        .input_src(&target)
        .fps(fps as i32)
        .output_options(recap_gst::record_window::OutputOptions {
            width: calc_width.into(),
            height: HEIGHT.into(),
            path: video_path.clone(),
        });

    let mut inference_frame_id = 0;
    let should_check_frame_for_pressed = AtomicBool::new(true);

    let on_new_data = {
        let frame_count = frame_count.clone();
        let time_since_last_frame = Mutex::new(Instant::now());
        let new_data_histogram = histogram!(NEW_DATA_INTERVAL, "id" => id.to_string());
        move || {
            let last_value = frame_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            {
                let mut time_since_last_frame = time_since_last_frame.lock().unwrap();
                let elapsed = time_since_last_frame.elapsed();
                // skip first frame as there is always a big delay
                if last_value != 0 {
                    new_data_histogram.record(elapsed.as_secs_f64());
                }
                *time_since_last_frame = Instant::now();
            }
            let input_frame = InputFrame::get_codes();
            // if there are > 10 keys pressed, log an error
            if input_frame.user_keys.len() > 10 {
                error!("Too many keys pressed: {:?}", input_frame);
            }

            // Initial frame check to see if bad keys are pressed
            if should_check_frame_for_pressed.swap(false, std::sync::atomic::Ordering::Relaxed)
                && (input_frame
                    .user_keys
                    .contains(&input_codes::Keycode::KpEqual)
                    || input_frame
                        .user_keys
                        .contains(&input_codes::Keycode::KpComma)
                    || input_frame.user_keys.contains(&input_codes::Keycode::Equal)
                    || input_frame.user_keys.contains(&input_codes::Keycode::Comma))
            {
                error!(
                    "Equal or comma keys pressed at the same time on start!!! {:?}",
                    input_frame
                );
                send_error(
                    id,
                    Some("Equal or comma key pressed at the same time on start!!!".to_string()),
                );
                FileSource::CommaEqualOnStartError.play();
            }

            let mut state = data_input_state.lock().unwrap();

            state.push(input_frame);
        }
    };

    // when new data has enterer the pipeline capture the current inputs
    let pipeline_builder = pipeline_builder
        .on_event({
            let encodering_histogram = histogram!(ENCODING_LATENCY, "id" => id.to_string());
            let last_time = parking_lot::Mutex::new(None);
            move |event| match event {
                recap_gst::record_window::Event::NewData => on_new_data(),
                recap_gst::record_window::Event::FinishedEncoding(timestamp) => {
                    let old_timestamp = std::mem::replace(&mut *last_time.lock(), timestamp);
                    if let (Some(old), Some(new)) = (old_timestamp, timestamp) {
                        let elapsed = new.saturating_sub(old);
                        let elapsed_secs = elapsed.seconds_f64();
                        encodering_histogram.record(elapsed_secs);
                    }
                }
            }
        })
        .audio_input(audio_device)
        .audio_volume(mic_volume)
        .enable_inference(cfg!(feature = "inference") && with_inference.is_some())
        // run when there is a new frame in the inference pipeline to consume
        .with_inference_callback({
            let mut time_since_last_frame = Instant::now();
            let inference_frame_interval =
                histogram!("inference_frame_interval", "id" => id.to_string());
            move |appsink, sample| {
                let buffer = sample.buffer().ok_or_else(|| {
                    gst::element_error!(
                        appsink,
                        gst::ResourceError::Failed,
                        ("Failed to get buffer from appsink")
                    );

                    gst::FlowError::Error
                })?;

                let map = buffer.map_readable().map_err(|_| {
                    gst::element_error!(
                        appsink,
                        gst::ResourceError::Failed,
                        ("Failed to map buffer readable")
                    );

                    gst::FlowError::Error
                })?;

                let pad = appsink.static_pad("sink").ok_or(gst::FlowError::Error)?;

                let caps = pad.current_caps().ok_or(gst::FlowError::Error)?;
                let s = caps.structure(0).ok_or(gst::FlowError::Error)?;
                let width = s.get::<i32>("width").map_err(|_| gst::FlowError::Error)?;
                let height = s.get::<i32>("height").map_err(|_| gst::FlowError::Error)?;

                let size = width * height * 3;
                let output = map.to_vec();
                if output.len() != size as usize {
                    FileSource::InferenceFailed.play();
                    panic!("Output size is not correct: {} != {}", output.len(), size);
                }

                if inference_frame_id != 0 {
                    let elapsed = time_since_last_frame.elapsed();
                    inference_frame_interval.record(elapsed.as_secs_f64());
                    time_since_last_frame = Instant::now();
                }

                inference_frame_id += 1;

                let output = match resize_image_core(
                    &output,
                    height as u32,
                    width as u32,
                    MODEL_INPUT_HEIGHT,
                    MODEL_INPUT_WIDTH,
                ) {
                    Ok(resized) => resized,
                    Err(err) => {
                        error!("Failed to resize image: {:?}", err);
                        return Err(gst::FlowError::Error);
                    }
                };

                let frame = video_inference_grpc::video_inference::Frame {
                    height: MODEL_INPUT_HEIGHT as i32,
                    width: MODEL_INPUT_WIDTH as i32,
                    data: output,
                    id: inference_frame_id,
                };
                inference_sender.send(frame).unwrap_or_else(|_| {
                    error!("Inference channel closed, dropping frame");
                });

                Ok(())
            }
        });

    // build the pipeline
    let pipeline = pipeline_builder.build()?;

    let stopper = pipeline.eos_sender();

    #[cfg(feature = "trace")]
    let trace_timeline = path.join("timeline.txt");

    // tell gstreamer to stop the pipeline when the stop signal is sent
    let join_handle = handle.spawn(async move {
        let start = std::time::Instant::now();

        // wait for the stop signal to be sent
        stop_capture_notify.notified().await;

        // save full timeline
        #[cfg(feature = "trace")]
        std::thread::spawn(move || {
            let Ok(mut file) = std::fs::File::create(&trace_timeline) else {
                error!("Failed to create trace file");
                return;
            };
            let full_events = crate::input_manager::timeline::TIMELINE
                .lock()
                .drain_full_events();
            for (i, event) in full_events.iter().enumerate() {
                std::io::Write::write_all(&mut file, format!("{}: {:?}\n", i, event).as_bytes())
                    .unwrap_or_else(|e| {
                        error!("Failed to write to trace file: {:?}", e);
                    });
            }
        });
        if with_inference.is_some() {
            // stop the inference stream
            stop_inference.notify_one();
        }
        debug!("Stopping capture");

        // stop the pipeline
        if let Err(err) = stopper.end() {
            error!("Error stopping pipeline: {:?}", err);
        }
        debug!("Stopped capture");

        start.elapsed()
    });

    info!(
        "Starting capture in version {}-{}]",
        RECAP_VERSION, GIT_COMMIT
    );

    // make sure key state is correct
    crate::input_manager::double_check_key_state();

    let start_time = crate::input_manager::reset_recording();

    let state = pipeline.run_till_end().expect("failed to run pipeline");

    info!("Pipeline stats: {:#?}", state);
    let duration = handle.block_on(join_handle).expect("failed to join handle");
    debug!("Pipeline finished");
    set_inference_running(None);

    // release all keys
    lift_simulated_keys();

    double_beep();

    let mut input_state = { input_state.lock().unwrap().clone() };

    let annotations_len = input_state.len();
    info!("Annotations length: {}", annotations_len);
    let audio_times = handle
        .block_on(audio_key)
        .expect("failed to join audio key handle");
    info!("Audio key presses length: {}", audio_times.len());

    let start = std::time::Instant::now();

    // some times the first events is before the start time, this is unexpected but can happen because of the spinlock mutex
    // to avoid this we will remove the first event if it is before the start time but if two events are before the start time we will log an error
    if let Some(frame) = input_state.first()
        && let Some(event) = frame.timeline.first()
        && event.time < start_time
    {
        warn!(
            "First input state time {:?} is before start time {:?}. removing first input state.",
            event.time, start_time
        );
        input_state[0].timeline.remove(0);
        if let Some(frame) = input_state.first()
            && let Some(event) = frame.timeline.first()
            && event.time < start_time
        {
            error!(
                "First input state time {:?} is still before start time {:?}. This is unexpected.",
                event.time, start_time
            );
            send_error(
                id,
                Some(format!(
                    "First input state time {:?} is still before start time {:?}. This is unexpected.",
                    event.time, start_time
                )),
            );
        }
    }

    on_finish_check::on_finish_check(
        id,
        annotations_len as u32,
        &video_path,
        frame_count.load(std::sync::atomic::Ordering::SeqCst),
        &input_state,
        fps,
        duration,
        start_time,
    )?;

    save_input_state(input_state, &path, meta_data, start_time, audio_times)?;
    trace!("took {:?} to finish", start.elapsed());
    info!(
        "Finished capturing saved to {}",
        path.as_os_str().to_str().unwrap()
    );

    Ok(())
}

fn next_multiple_of(start: i32, rhs: i32) -> i32 {
    // This would otherwise fail when calculating `r` when self == T::MIN.
    if rhs == -1 {
        return start;
    }

    let r = start % rhs;
    let m = if (r > 0 && rhs < 0) || (r < 0 && rhs > 0) {
        r + rhs
    } else {
        r
    };

    if m == 0 { start } else { start + (rhs - m) }
}

// WARNING: THE BELOW RESIZE IMAGE FUNCTION IS COPY PASTED FROM https://github.com/elefant-ai/elefant_rust. IF YOU
// MAKE CHANGES HERE, MAKE SURE TO ALSO UPDATE THE ORIGINAL REPO.
/// Core resize functionality that can be used from both Rust and Python
pub fn resize_image_core(
    image_data: &[u8],
    src_height: u32,
    src_width: u32,
    dst_height: u32,
    dst_width: u32,
) -> Result<Vec<u8>, String> {
    // Print warning if compiled in debug mode
    #[cfg(debug_assertions)]
    {
        eprintln!(
            "WARNING: image_resize module was compiled in debug mode. Performance may be significantly lower. Compile with --release for production."
        );
    }
    // Example check (optional but good practice):
    let expected_len = (src_width * src_height * 3) as usize; // Assuming U8x3
    if image_data.len() != expected_len {
        return Err(format!(
            "Input data length {} does not match dimensions {}x{}x3",
            image_data.len(),
            src_width,
            src_height
        ));
    }

    // Create an image reference
    let src_image = ImageRef::new(src_width, src_height, image_data, PixelType::U8x3)
        .map_err(|e| format!("Failed to create source image: {e}"))?;

    // Create a new image for the destination
    let mut dst_image = Image::new(dst_width, dst_height, PixelType::U8x3);

    // Create a resizer
    let mut resizer = Resizer::new();
    #[cfg(target_arch = "x86_64")]
    #[allow(unsafe_code)]
    unsafe {
        resizer.set_cpu_extensions(CpuExtensions::Avx2);
    }

    let resize_options =
        ResizeOptions::new().resize_alg(ResizeAlg::Interpolation(FilterType::Hamming));

    resizer
        .resize(&src_image, &mut dst_image, Some(&resize_options))
        .map_err(|e| format!("Failed to resize image: {e}"))?;

    // Extract the buffer
    Ok(dst_image.buffer().to_vec())
}
