use std::sync::Once;

static FFMPEG_INIT: Once = Once::new();

pub fn setup() {
    FFMPEG_INIT.call_once(|| {
        ffmpeg::init().unwrap();
    });
}

/// Decode a video file and call the provided callback for each decoded frame.
pub fn decode_video<F>(video_path: &str, mut new_frame_callback: F) -> Result<(), anyhow::Error>
where
    F: FnMut(ffmpeg::frame::Video),
{
    setup();

    let mut input = ffmpeg::format::input(&video_path)?;

    let video_stream = input
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or_else(|| anyhow::anyhow!("No video stream found"))?;

    let video_stream_index = video_stream.index();

    let context_decoder =
        ffmpeg::codec::context::Context::from_parameters(video_stream.parameters())?;
    let mut decoder = context_decoder.decoder().video()?;

    let mut scaler = ffmpeg::software::scaling::Context::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        ffmpeg::format::Pixel::RGB24,
        decoder.width(),
        decoder.height(),
        ffmpeg::software::scaling::Flags::BILINEAR,
    )?;

    let mut receive_and_process_decoded_frames =
        |decoder: &mut ffmpeg::decoder::Video| -> Result<(), ffmpeg::Error> {
            let mut decoded = ffmpeg::frame::Video::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                let mut rgb_frame = ffmpeg::frame::Video::empty();
                scaler.run(&decoded, &mut rgb_frame)?;
                new_frame_callback(rgb_frame);
            }
            Ok(())
        };

    for (stream, packet) in input.packets() {
        if stream.index() == video_stream_index {
            decoder.send_packet(&packet)?;
            receive_and_process_decoded_frames(&mut decoder)?;
        }
    }

    decoder.send_eof()?;
    receive_and_process_decoded_frames(&mut decoder)?;
    Ok(())
}
