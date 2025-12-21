use std::{str::FromStr, time::Duration};

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use video_annotation_proto::video_annotation::{VideoAnnotation, input_event};
use video_inference_grpc::prost::Message;

const PROTO_PATHS: &[&str] = &[
    "tests/assets/system_and_user.proto",
    "tests/assets/system_only.proto",
    "tests/assets/user_only.proto",
];

#[test]
fn basic() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        rdev::listen(move |event| {
            if let rdev::EventType::KeyPress(key) = event.event_type {
                if key == rdev::Key::Return {
                    tx.send(()).expect("Failed to send message");
                }
            }
        })
        .unwrap();
    });
    rdev::simulate(&rdev::EventType::KeyPress(rdev::Key::Return))
        .expect("Failed to simulate key press");
    rx.recv_timeout(std::time::Duration::from_secs(1))
        .expect("Failed to receive message");
}

#[ignore = "presses keys"]
#[tokio::test]
async fn playback_events() -> anyhow::Result<()> {
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    let path = PROTO_PATHS[2];

    let file = VideoAnnotation::decode(tokio::fs::read(path).await?.as_slice())?;
    let timeline: Vec<(u64, &input_event::Event)> = file
        .frame_annotations
        .par_iter()
        .flat_map(|v| &v.input_events)
        .filter_map(|event| event.event.as_ref().map(|e| (event.time, e)))
        .filter(|(_, e)| {
            !matches!(
                e,
                // should be able to use mouse move
                input_event::Event::MouseMoveEvent(_)
                    | input_event::Event::MouseDeltaEvent(_)
                    | input_event::Event::GamePadAxisEvent(_)
                    | input_event::Event::GamePadButtonEvent(_)
                    | input_event::Event::GamePadTriggerEvent(_)
            )
        })
        .collect();
    let total_annos = timeline.len();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

    std::thread::spawn(move || {
        let mut count = 0;
        rdev::listen(move |event| {
            count += 1;
            if count % 100 == 0 {
                println!("Processed {count} events");
            }
            tx.send(event).expect("Failed to send event");
        })
        .unwrap();
    });

    println!("Sending {} events", timeline.len());

    for (_, event) in &timeline {
        let event = match event {
            input_event::Event::MouseEvent(mouse_button_event) => {
                if mouse_button_event.pressed {
                    rdev::EventType::ButtonPress(
                        input_codes::Button::from_str(&mouse_button_event.button)
                            .unwrap()
                            .into(),
                    )
                } else {
                    rdev::EventType::ButtonRelease(
                        input_codes::Button::from_str(&mouse_button_event.button)
                            .unwrap()
                            .into(),
                    )
                }
            }
            input_event::Event::KeyboardEvent(keyboard_event) => {
                if keyboard_event.pressed {
                    rdev::EventType::KeyPress(
                        input_codes::Keycode::from_str(&keyboard_event.key)
                            .unwrap()
                            .try_into()
                            .unwrap(),
                    )
                } else {
                    rdev::EventType::KeyRelease(
                        input_codes::Keycode::from_str(&keyboard_event.key)
                            .unwrap()
                            .try_into()
                            .unwrap(),
                    )
                }
            }
            input_event::Event::MouseMoveEvent(vec2_int) => rdev::EventType::MouseMove {
                x: vec2_int.x as f64,
                y: vec2_int.y as f64,
            },
            input_event::Event::WheelEvent(vec2_int) => rdev::EventType::Wheel {
                delta_x: vec2_int.x as i64,
                delta_y: vec2_int.y as i64,
            },
            _ => continue,
        };
        rdev::simulate(&event).expect("Failed to simulate event");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    println!("done sending events");

    let mut output: Vec<input_event::Event> = Vec::with_capacity(total_annos);

    while let Ok(Some(event)) = tokio::time::timeout(Duration::from_millis(500), rx.recv()).await {
        let event = match event.event_type {
            rdev::EventType::KeyPress(key) => input_event::Event::KeyboardEvent(
                video_annotation_proto::video_annotation::KeyboardEvent {
                    key: input_codes::Keycode::from(key).to_string(),
                    pressed: true,
                },
            ),
            rdev::EventType::KeyRelease(key) => input_event::Event::KeyboardEvent(
                video_annotation_proto::video_annotation::KeyboardEvent {
                    key: input_codes::Keycode::from(key).to_string(),
                    pressed: false,
                },
            ),
            rdev::EventType::ButtonPress(button) => input_event::Event::MouseEvent(
                video_annotation_proto::video_annotation::MouseButtonEvent {
                    button: input_codes::Button::from(button).to_string(),
                    pressed: true,
                },
            ),
            rdev::EventType::ButtonRelease(button) => input_event::Event::MouseEvent(
                video_annotation_proto::video_annotation::MouseButtonEvent {
                    button: input_codes::Button::from(button).to_string(),
                    pressed: false,
                },
            ),
            rdev::EventType::MouseMove { x, y } => {
                input_event::Event::MouseMoveEvent(glam::ivec2(x as i32, y as i32).into())
            }
            rdev::EventType::Wheel { delta_x, delta_y } => {
                input_event::Event::WheelEvent(glam::ivec2(delta_x as i32, delta_y as i32).into())
            }
        };
        output.push(event);
    }

    if output.len() != total_annos {
        panic!("Expected {} events, but got {}", total_annos, output.len());
    }

    if output.len() != timeline.len() {
        panic!(
            "Expected {} events in timeline, but got {}",
            timeline.len(),
            output.len()
        );
    }

    let errors = output
        .par_iter()
        .zip(timeline)
        .filter_map(|(a, (_, b))| {
            if a != b {
                Some(format!("Mismatch : {a:?} != {b:?}"))
            } else {
                None
            }
        })
        .collect_vec_list();

    if !errors.is_empty() {
        let len = errors.len();
        eprintln!("Found {len} errors:");
        for error in errors.iter().flatten() {
            eprintln!(" - {error}");
        }
        anyhow::bail!("Playback events did not match the expected timeline. Found {len} errors.");
    }

    Ok(())
}
