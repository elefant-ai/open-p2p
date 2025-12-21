use std::{collections::HashSet, time::Duration};

use anyhow::Context;
use glam::IVec2;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator as _};
use tracing::{error, trace, warn};

use crate::{
    external::send_error,
    input_manager::{Event, HOT_KEYS},
};
use recap_gst::video_checks::{get_first_video_stream_info, get_real_frame_count};

use super::input::InputFrame;

#[allow(clippy::too_many_arguments)]
pub fn on_finish_check(
    id: uuid::Uuid,
    annotations_len: u32,
    video_path: impl AsRef<std::path::Path>,
    frame_count: u32,
    annotations: &[InputFrame],
    wanted_fps: f64,
    duration: Duration,
    start_time: std::time::SystemTime,
) -> Result<(), anyhow::Error> {
    let (video_info, video_stream_info) =
        get_first_video_stream_info(&video_path).context("get_first_video_stream_info")?;
    let fps = video_stream_info.framerate;
    let actual_frame_count = get_real_frame_count(&video_path).context("get_actual_frame_count")?;

    let min_fps = wanted_fps - 1.0;
    let max_fps = wanted_fps + 1.0;

    if fps < min_fps || fps > max_fps {
        send_error(id, Some(format!("FPS mismatch: expected 20 but got {fps}")));
        error!("FPS mismatch: expected 20 but got {}", fps);
    }

    trace!("Actual FPS: {}", fps);

    if actual_frame_count != frame_count {
        send_error(
            id,
            Some(format!(
                "Frame count mismatch: expected {frame_count} but got {actual_frame_count}"
            )),
        );
        error!(
            "Frame count mismatch: expected {} but got {} in actual frame count",
            frame_count, actual_frame_count
        );
    }

    trace!("Actual frame count: {}", actual_frame_count);

    if actual_frame_count != annotations_len {
        send_error(
            id,
            Some(format!(
                "Annotation count mismatch: expected {actual_frame_count} but got {annotations_len}"
            )),
        );
        error!(
            "Annotation count mismatch: expected {} but got {}",
            actual_frame_count, annotations_len
        );
    }

    trace!(
        "Duration: {:#?} Video info duration: {:#?}",
        duration, video_info.duration
    );

    let time_based_frame_count = (duration.as_secs_f64() * wanted_fps).round();
    const LENIENCY: f64 = 0.02; // 2% leniency
    // figure out the percentage on the time-based frame count
    let min_time_based_frame_count =
        (time_based_frame_count - (time_based_frame_count * LENIENCY)).round() as u32;
    let max_time_based_frame_count =
        (time_based_frame_count + (time_based_frame_count * LENIENCY)).round() as u32;

    if actual_frame_count < min_time_based_frame_count
        || actual_frame_count > max_time_based_frame_count
    {
        send_error(
            id,
            Some(format!(
                "Time-based frame count mismatch: expected near {time_based_frame_count} but got {actual_frame_count}"
            )),
        );
        error!(
            "Time-based frame count mismatch: expected near {} but got {}",
            time_based_frame_count, actual_frame_count
        );
    }

    check_frame_overlap(annotations, id);
    check_frame_user_and_keys(annotations, id);
    check_timeline(annotations, id, start_time);

    Ok(())
}

fn check_timeline(frames: &[InputFrame], id: uuid::Uuid, start_time: std::time::SystemTime) {
    // set starting keys as events wont cover these because its before the recording started. i.e. the recording is started with w pressed so there is no w press event only release
    let mut keys = frames
        .first()
        .map(|f| {
            let mut keys = HashSet::new();
            for event in &f.user_keys {
                keys.insert(event.clone());
            }
            keys
        })
        .unwrap_or_default();

    let mut simulated_keys = frames
        .first()
        .map(|f| {
            let mut simulated_keys = HashSet::new();
            for event in &f.system_keys {
                simulated_keys.insert(event.clone());
            }
            simulated_keys
        })
        .unwrap_or_default();
    let mut mouse_buttons = frames
        .first()
        .map(|f| {
            let mut mouse_buttons = HashSet::new();
            for event in &f.user_mouse.buttons {
                mouse_buttons.insert(*event);
            }
            mouse_buttons
        })
        .unwrap_or_default();

    let mut simulated_mouse_buttons = frames
        .first()
        .map(|f| {
            let mut simulated_mouse_buttons = HashSet::new();
            for event in &f.system_mouse.buttons {
                simulated_mouse_buttons.insert(*event);
            }
            simulated_mouse_buttons
        })
        .unwrap_or_default();

    let mut set_mouse_pos = None;

    let mut current_inference_running =
        frames.first().map(|f| f.inference_running).unwrap_or(false);

    let mut after_change_reduce_strictness = 0;

    frames.iter().enumerate().for_each(|(i, frame)| {
        let inference_changed = frame.inference_running != current_inference_running;
        current_inference_running = frame.inference_running;
        if inference_changed {
            after_change_reduce_strictness = 5; // for the next 5 frames reduce strictness
        } else if after_change_reduce_strictness > 0 {
            after_change_reduce_strictness -= 1;
        }

        let start = i
            .checked_sub(1)
            .map(|i| frames[i].time)
            .unwrap_or(start_time);

        // check each event is before the current frame and after the last frame
        let count = frame
            .timeline
            .iter()
            .filter(|event| {
                event.time < start || event.time > frame.time
            })
            .count();
        // one can commonly be out due to timeing and mutex
        if count > 0 {
            // send_error(
            //     id,
            //     Some(format!(
            //         "Timeline events out of order at frame {}: {} events found",
            //         i, count
            //     )),
            // );
            warn!(
                "Timeline events out of order at frame {}: {} events found",
                i, count
            );
        }

        // check mouse events match frame delta
        let mouse_delta = frame
            .timeline
            .iter()
            .filter_map(|e| {
                if let Event::MouseDelta(change) = &e.event {
                    Some(*change)
                } else {
                    None
                }
            })
            .sum::<IVec2>();

        let user_system_delta = frame.user_mouse.delta + frame.system_mouse.delta;

        if mouse_delta != user_system_delta {
            send_error(
                id,
                Some(format!(
                    "Mouse delta mismatch at frame {i}: expected {user_system_delta:?} but got {mouse_delta:?}"
                )),
            );
            error!(
                "Mouse delta mismatch at frame {}: expected {:?} but got {:?}",
                i, user_system_delta, mouse_delta
            );
        }

        // sum scroll change and match to frame
        let mouse_scroll = frame
            .timeline
            .iter()
            .filter_map(|e| {
                if let Event::MouseWheel(change) = &e.event {
                    Some(*change)
                } else {
                    None
                }
            })
            .sum::<IVec2>();

        let user_system_scroll = frame.user_mouse.scroll + frame.system_mouse.scroll;

        if mouse_scroll != user_system_scroll {
            send_error(
                id,
                Some(format!(
                    "Mouse scroll mismatch at frame {i}: expected {user_system_scroll:?} but got {mouse_scroll:?}"
                )),
            );
            error!(
                "Mouse scroll mismatch at frame {}: expected {:?} but got {:?}",
                i, user_system_scroll, mouse_scroll
            );
        }

        // check mouse position matches the last mouse move event
        let mouse_pos = frame
            .timeline
            .iter()
            .filter_map(|e| {
                if let Event::MouseMove(pos) = &e.event {
                    Some(*pos)
                } else {
                    None
                }
            })
            .next_back();

        let user_system_mouse_pos = frame.user_mouse.mouse_pos + frame.system_mouse.mouse_pos;

        if let Some(mouse_pos) = mouse_pos
            .inspect(|&pox| {
                set_mouse_pos = Some(pox);
            })
            .or(set_mouse_pos)
            && mouse_pos != user_system_mouse_pos
        {
            send_error(
                id,
                Some(format!(
                    "Mouse position mismatch at frame {i}: expected {user_system_mouse_pos:?} but got {mouse_pos:?}"
                )),
            );
            error!(
                "Mouse position mismatch at frame {}: expected {:?} but got {:?}",
                i, user_system_mouse_pos, mouse_pos
            );
        }

        frame.timeline.iter().for_each(|e| {
            if let Event::MouseButton { pressed, button } = &e.event {
                if e.simulated {
                    if *pressed {
                        simulated_mouse_buttons.insert(*button);
                    } else {
                        simulated_mouse_buttons.remove(button);
                    }
                } else if *pressed {
                    mouse_buttons.insert(*button);
                } else {
                    mouse_buttons.remove(button);
                }
            }
        });

        // Check user buttons separately
        let mut user_buttons = frame.user_mouse.buttons.clone();
        user_buttons.dedup();
        user_buttons.sort();

        let mut tracked_user_buttons = mouse_buttons.clone().into_iter().collect::<Vec<_>>();
        tracked_user_buttons.dedup();
        tracked_user_buttons.sort();

        if tracked_user_buttons != user_buttons {
            send_error(
                id,
                Some(format!(
                    "User mouse buttons mismatch at frame {i}: expected {user_buttons:?} but got {tracked_user_buttons:?}"
                )),
            );
            error!(
                "User mouse buttons mismatch at frame {}: expected {:?} but got {:?}",
                i, user_buttons, tracked_user_buttons
            );
        }

        // Check system buttons separately
        let mut system_buttons = frame.system_mouse.buttons.clone();
        system_buttons.dedup();
        system_buttons.sort();

        let mut tracked_simulated_buttons = simulated_mouse_buttons
            .clone()
            .into_iter()
            .collect::<Vec<_>>();
        tracked_simulated_buttons.dedup();
        tracked_simulated_buttons.sort();

        if tracked_simulated_buttons != system_buttons {
            if after_change_reduce_strictness == 0
            // only send error if not just after an inference state change
            {
                send_error(
                    id,
                    Some(format!(
                        "System mouse buttons mismatch at frame {i}: expected {system_buttons:?} but got {tracked_simulated_buttons:?}"
                    )),
                );
            }
            error!(
                "System mouse buttons mismatch at frame {}: expected {:?} but got {:?}",
                i, system_buttons, tracked_simulated_buttons
            );
        }

        frame.timeline.iter().for_each(|e| {
            if let Event::KeyboardInput { pressed, key } = &e.event {
                if HOT_KEYS.contains(key) {
                    return;
                }
                if e.simulated {
                    if *pressed {
                        simulated_keys.insert(key.clone());
                    } else {
                        simulated_keys.remove(key);
                    }
                } else if *pressed {
                    keys.insert(key.clone());
                } else {
                    keys.remove(key);
                }
            }
        });

        // Check user keys separately
        let mut user_keys = frame.user_keys.clone();
        user_keys.dedup();
        user_keys.sort();

        let mut tracked_user_keys = keys.clone().into_iter().collect::<Vec<_>>();
        tracked_user_keys.dedup();
        tracked_user_keys.sort();

        if tracked_user_keys != user_keys {
            send_error(
                id,
                Some(format!(
                    "User keys mismatch at frame {i}: expected {user_keys:?} but got {tracked_user_keys:?}"
                )),
            );
            error!(
                "User keys mismatch at frame {}: expected {:?} but got {:?}",
                i, user_keys, tracked_user_keys
            );
        }

        // Check system keys separately
        let mut system_keys = frame.system_keys.clone();
        system_keys.dedup();
        system_keys.sort();

        let mut tracked_simulated_keys = simulated_keys.clone().into_iter().collect::<Vec<_>>();
        tracked_simulated_keys.dedup();
        tracked_simulated_keys.sort();

        if tracked_simulated_keys != system_keys {
            if after_change_reduce_strictness == 0
            // only send error if not just after an inference state change
            {
                send_error(
                    id,
                    Some(format!(
                        "System keys mismatch at frame {i}: expected {system_keys:?} but got {tracked_simulated_keys:?}"
                    )),
                );
            }
            error!(
                "System keys mismatch at frame {}: expected {:?} but got {:?}",
                i, system_keys, tracked_simulated_keys
            );
        }
    });
}

/// check for frames where user keys are not empty when inference is running
/// and system keys are not empty when inference is not running
fn check_frame_user_and_keys(frames: &[InputFrame], id: uuid::Uuid) {
    frames.par_iter().enumerate().for_each(|(i, frame)| {
        if frame.inference_running {
            if !frame.user_keys.is_empty() {
                send_error(
                    id,
                    Some(format!(
                        "User keys are not empty when inference is running on frame {}: {:?}",
                        i, frame.user_keys
                    )),
                );
                error!(
                    "User keys are not empty when inference is running on frame {}: {:?}",
                    i, frame.user_keys
                );
            }
        } else if !frame.system_keys.is_empty() {
            send_error(
                id,
                Some(format!(
                    "System keys are not empty when inference is not running on frame {}: {:?}",
                    i, frame.system_keys
                )),
            );
            error!(
                "System keys are not empty when inference is not running on frame {}: {:?}",
                i, frame.system_keys
            );
        }
    });
}

/// check for frames where keys in user and system overlap
fn check_frame_overlap(frames: &[InputFrame], id: uuid::Uuid) {
    let frames_that_overlap = frames
        .par_iter()
        .enumerate()
        .filter_map(|(i, frame)| {
            if !frame.user_keys.is_empty() && !frame.system_keys.is_empty() {
                Some((i, frame))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let frames_overlay = frames_that_overlap.len();

    if frames_overlay > 0 {
        tracing::error!(
            "There are {} frames that overlap with other frames",
            frames_overlay
        );
        send_error(
            id,
            Some(format!(
                "There are {frames_overlay} frames that overlap with other frames"
            )),
        );
        for (i, frame) in frames_that_overlap {
            tracing::error!("Frame {}: {:?}", i, frame);
        }
    }
}
