"""Thin wrapper around ffmpeg to encode videos."""

import subprocess
import torch
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional


class FFMpegEncoderSettings:
    pass


class FastFFmpegEncoderSettings(FFMpegEncoderSettings):
    def _encoder_settings(self):
        return [
            "-c:v",
            "libx264",  # Faster codec: H.264
            "-preset",
            "fast",  # Fastest encoding preset
            "-crf",
            "23",  # Adjust quality (lower = better quality, larger file)
        ]


@dataclass
class NvidiaFFmpegEncoderSettings(FFMpegEncoderSettings):
    """
    Hardware-accelerated H.264 encoding via NVIDIA NVENC.
    """

    preset: Optional[str] = None
    qp: Optional[str] = None

    def _encoder_settings(self):
        if self.qp is not None:
            qp = self.qp
        else:
            qp = np.random.randint(6, 18)
        if self.preset is None:
            self.preset = "medium"

        encode_settings = [
            "-c:v",
            "h264_nvenc",
            "-preset",
            self.preset,
            "-qp",
            f"{qp}",
        ]
        return encode_settings


@dataclass
class LowResFFmpegEncoderSettings(FFMpegEncoderSettings):
    """
    Near-loss-less H.264. Produces files that are visually identical to source but
    much smaller than true loss-less. Use for training data that must keep detail.
    """

    encode_color_space: str
    preset: Optional[str] = None
    crf: Optional[str] = None
    use_fast_decode: Optional[bool] = None

    def _encoder_settings(self):
        if self.crf is not None:
            crf = self.crf
        else:
            crf = np.random.randint(6, 18)
        if self.use_fast_decode is not None:
            use_fast_decode = self.use_fast_decode
        else:
            use_fast_decode = np.random.random() < 0.5
        if self.preset is None:
            self.preset = "medium"
        if self.encode_color_space == "yuv":
            encode_str = "libx264"
        elif self.encode_color_space == "rgb":
            encode_str = "libx264rgb"
        else:
            raise ValueError(f"Invalid encode color space: {self.encode_color_space}")

        encode_settings = [
            "-c:v",
            encode_str,
            "-preset",
            self.preset,
            "-crf",
            f"{crf}",
        ]

        if use_fast_decode:
            return encode_settings + ["-tune", "fastdecode"]
        else:
            return encode_settings


class FFmpegEncoder:
    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        encoder_settings: FFMpegEncoderSettings,
        use_cuda: bool = False,
    ):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = 20

        if self.width % 2 != 0:
            # Encoder requires even width
            self._frame_width_odd = True
            logging.warning(f"Frame width {self.width} is not even, truncating 1 pixel")
            self.width -= 1
        else:
            self._frame_width_odd = False

        if use_cuda:
            ffmpeg_config = [
                "/opt/elefant/ffmpeg/ffmpeg",
                "-hwaccel",
                "cuda",
            ]
        else:
            ffmpeg_config = [
                "ffmpeg",
            ]
        self.ffmpeg_command = (
            [
                *ffmpeg_config,
                "-y",  # Overwrite output file without asking
                "-loglevel",
                "warning",  # Only show errors, suppress warnings and info messages
                "-f",
                "rawvideo",  # Input format
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "bgr24",  # Input pixel format (we'll send BGR)
                "-s",
                f"{self.width}x{self.height}",  # Input size
                "-r",
                str(self.fps),  # Input frame rate
                "-i",
                "-",  # Input comes from pipe (stdin)
            ]
            + encoder_settings._encoder_settings()
            + [
                "-pix_fmt",
                "yuv420p",  # Output pixel format for compatibility when using yuv encoder, will be ignored if using rgb encoder
                self.output_path,  # Output file path
            ]
        )

    def __enter__(self):
        self.ffmpeg_process = subprocess.Popen(
            self.ffmpeg_command,
            stdin=subprocess.PIPE,
            # Don't capture stdout or stderr, that could block.
            stdout=None,
            stderr=None,
        )
        self.n_frames = 0
        return self

    def encode_frame(self, frame: torch.Tensor):
        if self._frame_width_odd:
            frame = frame[:, :, :-1]
        assert frame.shape == (3, self.height, self.width)
        assert frame.dtype == torch.uint8
        frame = frame.permute(1, 2, 0).numpy()
        # Change to BGR
        frame = frame[:, :, [2, 1, 0]]
        self.ffmpeg_process.stdin.write(frame.tobytes())
        self.n_frames += 1

    def __exit__(self, exc_type, exc_value, traceback):
        self.ffmpeg_process.stdin.close()
        self.ffmpeg_process.wait()
