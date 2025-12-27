from typing import List, Optional
import argparse
from elefant.config import load_config, ConfigBase, WandbConfig
from elefant.data import boto
from elefant.launcher import LaunchConfig
import os
import datetime
import logging
import wandb
import pydantic
import tempfile
import concurrent
import dateutil.parser
import threading
from concurrent.futures import ThreadPoolExecutor
import traceback
import botocore
from boto3.s3.transfer import TransferConfig
from elefant.data.rescale.rescale import rescale_local_video


class RescaleConfig(ConfigBase):
    wandb: WandbConfig = pydantic.Field(default=WandbConfig())
    prefixes: List[str]
    frame_height: int
    frame_width: int
    n_threads: int = 1

    # If fps is set, recode the video to the given fps.
    fps: Optional[int] = None
    encode_type: Optional[str] = None
    encode_color_space: str = "yuv"
    quality_factor: Optional[str] = None
    use_fast_decode: Optional[bool] = None
    preset: Optional[str] = None

    # If last_modified is set, only rescale if the video was last modified before the given date.
    # This is useful for overwriting videos that have been rescaled previously.
    # after changes.
    overwrite_older_than: Optional[datetime.datetime] = None
    probability_of_nvidia_encoding: float = 0.9

    @pydantic.validator("overwrite_older_than", pre=True)
    def parse_overwrite_older_than_naive(cls, value):
        if value is None:
            return None
        if isinstance(value, datetime.datetime):
            return value.replace(
                tzinfo=datetime.timezone.utc
            )  # Make it naive by removing timezone
        if isinstance(value, datetime.date):
            return datetime.datetime.combine(value, datetime.time.min).replace(
                tzinfo=datetime.timezone.utc
            )
        if isinstance(value, str):
            return dateutil.parser.parse(value).replace(tzinfo=datetime.timezone.utc)
        raise ValueError(f"Invalid overwrite_older_than value: {value}")

    launch: LaunchConfig = pydantic.Field(default_factory=lambda: LaunchConfig())

    # Seems to be the only way to set default values for nested fields.
    # that aren't ignored if one field (e.g. name) is set in yaml.
    @pydantic.validator("launch", pre=False)
    def set_launch_defaults(cls, value):
        if value.job_name == "":
            value.job_name = "rescale"
        if value.vm_flavor == "":
            value.vm_flavor = "n3-L40x1"
        if value.run_cmd == "":
            value.run_cmd = (
                "uv run python elefant/data/rescale/rescale.py --config=%(config_path)s"
            )
        return value


class Rescaler:
    def __init__(self, config: RescaleConfig):
        self.config = config
        self._data_tempdir = os.environ.get(
            "ELEFANT_TMP_DATA_PATH", "/tmp/elefant_data"
        )
        os.makedirs(self._data_tempdir, exist_ok=True)
        logging.info(f"Using temp data dir: {self._data_tempdir}")

        self._data_bucket = boto.get_training_data_bucket(write=True)

        if self.config.n_threads > 1:
            self._thread_pool = ThreadPoolExecutor(max_workers=self.config.n_threads)

        if self.config.wandb.enabled:
            wandb.init(
                project=self.config.wandb.project,
                name=self.config.wandb.exp_name,
                tags=self.config.wandb.tags,
                config=self.config.model_dump(),
            )

        self._n_rescaled = 0
        self._n_skipped = 0
        self._n_errors = 0
        # Make a mutex for the wandb logging.
        self._wandb_lock = threading.Lock()

    def update_wandb_stats(self, skipped: bool = False, error: bool = False):
        with self._wandb_lock:
            if skipped:
                self._n_skipped += 1
            elif error:
                self._n_errors += 1
            else:
                self._n_rescaled += 1
            wandb.log(
                {
                    "n_rescaled": self._n_rescaled,
                    "n_skipped": self._n_skipped,
                    "n_errors": self._n_errors,
                }
            )

    def run(self):
        files_in_prefix = []
        for prefix in self.config.prefixes:
            files_in_prefix.extend(
                [o.key for o in self._data_bucket.objects.filter(Prefix=prefix)]
            )
        videos = sorted([v for v in files_in_prefix if "video." in v])

        futures = []
        for video_path in videos:
            if self.config.n_threads == 1:
                futures.append(self._check_and_rescale_video(video_path))
            else:
                futures.append(
                    self._thread_pool.submit(self._check_and_rescale_video, video_path)
                )

        if self.config.n_threads > 1:
            # Wait for all futures to complete and raise any exceptions
            for future in concurrent.futures.as_completed(futures):
                future.result()

    def _check_and_rescale_video(self, video_path: str):
        # First check if the video is already rescaled.
        ## TODO: remove this later
        if (
            self.config.quality_factor is not None
            and self.config.use_fast_decode is not None
            and self.config.preset is not None
            and self.config.encode_type is not None
            and self.config.encode_color_space is not None
        ):
            output_file = f"/{self.config.frame_height}x{self.config.frame_width}_{self.config.encode_type}_{self.config.encode_color_space}_{self.config.quality_factor}_{self.config.use_fast_decode}_{self.config.preset}.mp4"
        else:
            output_file = f"/{self.config.frame_height}x{self.config.frame_width}.mp4"
        rescaled_video_path = os.path.dirname(video_path) + output_file
        # Check if file exists in S3 without downloading
        try:
            head = self._data_bucket.meta.client.head_object(
                Bucket=self._data_bucket.name,
                Key=rescaled_video_path,
            )
            last_modified = head["LastModified"]

            if (
                self.config.overwrite_older_than
                and last_modified < self.config.overwrite_older_than
            ):
                self._rescale_video(video_path, rescaled_video_path)
                self.update_wandb_stats(skipped=False)
            else:
                logging.info(f"Video {video_path} already rescaled.")
                self.update_wandb_stats(skipped=True)
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                # File doesn't exist
                try:
                    self._rescale_video(video_path, rescaled_video_path)
                except Exception as e:
                    logging.error(f"Error rescaling video {video_path}: {e}")
                    traceback.print_exc()
                    self.update_wandb_stats(error=True)
            else:
                logging.exception(f"Error rescaling video {video_path}")
                self.update_wandb_stats(error=True)

    def _rescale_video(self, video_path: str, rescaled_video_path: str):
        logging.info(f"Rescaling video {video_path} to {rescaled_video_path}")

        # Download the video to a temp file.
        orig_ext = video_path.split(".")[-1]
        # Keep this in scope until the done reading.
        temp_video_file = tempfile.NamedTemporaryFile(
            suffix=f".{orig_ext}", dir=self._data_tempdir
        )
        temp_video_path = temp_video_file.name
        logging.info(f"Downloading video {video_path} to {temp_video_path}")
        self._data_bucket.download_file(
            video_path, temp_video_path, Config=TransferConfig(use_threads=False)
        )

        # Rescale/recode video.
        rescaled_video_tmp_file = tempfile.NamedTemporaryFile(
            suffix=".mp4", dir=self._data_tempdir
        )
        self._recode_video(temp_video_path, rescaled_video_tmp_file.name)

        # Upload the new video to R2
        self._data_bucket.upload_file(
            rescaled_video_tmp_file.name,
            rescaled_video_path,
            Config=TransferConfig(use_threads=False),
        )

        # Delete the temp files.
        temp_video_file.close()
        rescaled_video_tmp_file.close()

    def _recode_video(self, video_path: str, rescaled_video_path: str):
        rescale_local_video(
            video_path,
            self.config.frame_height,
            self.config.frame_width,
            output_path=rescaled_video_path,
            fps=self.config.fps,
            rescale_config=self.config,
            probability_of_nvidia_encoding=self.config.probability_of_nvidia_encoding,
        )


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(filename)s:%(lineno)d %(message)s"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config, RescaleConfig)
    Rescaler(config).run()


if __name__ == "__main__":
    main()
