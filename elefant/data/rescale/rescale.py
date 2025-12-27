from typing import Optional

try:
    from torchcodec.decoders import VideoDecoder
except RuntimeError:
    # Allow importing this file even if torchcodec is not installed.
    pass
from elefant.data.rescale.resize import resize_image_for_model
from elefant.ffmpeg import (
    FFmpegEncoder,
    LowResFFmpegEncoderSettings,
    NvidiaFFmpegEncoderSettings,
)
import logging
import time
import numpy as np


def get_encoder_settings(probability_of_nvidia_encoding: float):
    if np.random.random() > probability_of_nvidia_encoding:
        encode_type = "software"
        if np.random.random() < 0.5:
            logging.info(f"Using software encoding with yuv color space")
            encoder_settings = LowResFFmpegEncoderSettings(
                preset="medium",
                crf=None,
                use_fast_decode=None,
                encode_color_space="yuv",
            )
        else:
            logging.info(f"Using software encoding with rgb color space")
            encoder_settings = LowResFFmpegEncoderSettings(
                preset="medium",
                crf=None,
                use_fast_decode=None,
                encode_color_space="rgb",
            )
    else:
        logging.info(f"Using nvidia encoding")
        encoder_settings = NvidiaFFmpegEncoderSettings(
            preset="medium",
            qp=None,
        )
        encode_type = "nvidia"
    return encoder_settings, encode_type


def rescale_local_video(
    video_path: str,
    frame_height: int,
    frame_width: int,
    output_path: str,
    fps: Optional[int] = None,
    rescale_config=None,
    probability_of_nvidia_encoding: float = 0.9,
):
    # Use ffmpeg to rescale the video and encode a1 mp4.
    # TODO: use GPU/CUDA acceleration.
    fps_opt = []
    if fps:
        raise NotImplementedError("Constant frame rate is not implemented yet.")
        # Recode to the given fps at a constant frame rate.
        fps_opt = ["-fps_mode", "cfr", "-r", str(fps)]
    else:
        # Passthrough the fps.
        fps_opt = ["-fps_mode", "passthrough"]

    video_decoder = VideoDecoder(source=video_path, device="cpu")

    # Check the fps is close to 20.
    print(
        f"fps check: {video_path}, average_fps: {video_decoder.metadata.average_fps}, num_frames_from_content: {video_decoder.metadata.num_frames_from_content}, duration_seconds: {video_decoder.metadata.duration_seconds}"
    )
    assert video_decoder.metadata.average_fps >= 19.0
    assert video_decoder.metadata.average_fps <= 21.0
    encode_type = None
    if rescale_config is None:
        # use a mixture of encoding
        logging.info("No encode type specified, using a mixture of software and nvidia")
        encoder_settings, encode_type = get_encoder_settings(
            probability_of_nvidia_encoding
        )
    else:
        if rescale_config.encode_type == "software":
            encoder_settings = LowResFFmpegEncoderSettings(
                preset="medium"
                if rescale_config.preset is None
                else rescale_config.preset,
                crf=rescale_config.quality_factor,
                use_fast_decode=rescale_config.use_fast_decode,
                encode_color_space=rescale_config.encode_color_space,
            )
            encode_type = "software"
        elif rescale_config.encode_type == "nvidia":
            encoder_settings = NvidiaFFmpegEncoderSettings(
                preset="medium"
                if rescale_config.preset is None
                else rescale_config.preset,
                qp=rescale_config.quality_factor,
            )
            encode_type = "nvidia"
        elif rescale_config.encode_type is None:
            logging.info(
                "No encode type specified, using a mixture of software and nvidia"
            )
            encoder_settings, encode_type = get_encoder_settings(
                probability_of_nvidia_encoding
            )
        else:
            raise ValueError(f"Invalid encode type: {rescale_config.encode_type}")

    with FFmpegEncoder(
        output_path=output_path,
        width=frame_width,
        height=frame_height,
        encoder_settings=encoder_settings,
        use_cuda=True if encode_type == "nvidia" else False,
    ) as encoder:
        n_frames = len(video_decoder)
        start_time = time.time()
        last_log_time = start_time
        for i, frame in enumerate(video_decoder):
            # Log every periodically.
            if time.time() - last_log_time > 120.0:
                encoding_fps = i / (time.time() - start_time)
                logging.info(f"Encoding frame {i}/{n_frames}, {encoding_fps:.2f} fps")
                last_log_time = time.time()
            frame = resize_image_for_model(frame, (frame_height, frame_width))
            encoder.encode_frame(frame)
