"""
This is our key dataset class.

We need to read videos consecutively (for efficiency) but we want the data to be as close to IID as possible..
We also need to support multi-GPU training (DDP).

We solve this having the first dataset worker start a bunch of processes (to avoid GIL).

The processes are:
- There is a single queue process that continually queues local proto paths.
- There are multiple preprocess workers that preprocess the protos and videos into data for training.
- There is a single shuffle worker that shuffles the data.
- There are multiple dataset workers that get data from the shuffle worker.

The communication between the processes is done via zeromq queues (zmq_queue.py).

Note that this class is mainly used for streaming data from a remote storage, rather than local disk.
"""

import logging
import os
import torch.multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Any, Optional, List
from torch.utils.data import DataLoader
import numpy as np
import pydantic
import torch
import time
from elefant.data import zmq_queue
import elefant_rust
import uuid
import torch.nn.functional as F

try:
    from torchcodec.decoders import VideoDecoder
except Exception:
    # This lets us run on mac without setting DYLD_LIBRARY_PATH.
    # Will not be able to use the dataset if this import fails.
    VideoDecoder = None

from elefant.config import ConfigBase
from elefant.data.proto import video_annotation_pb2


class RandAugmentationConfig(ConfigBase):
    # The fraction of examples that will be augmented.
    fraction_augmented: float = 0.0
    augmentations: List[str] = []


class VideoProtoDatasetConfig(ConfigBase):
    local_prefix: str

    frame_height: int = 192
    frame_width: int = 192
    # Number of frames to return in each chunk.
    T: int = 10
    always_labelled: bool = False

    n_preprocess_workers_per_iter_worker: int = 8
    preprocessed_chunks_queue_size: int = 1024

    shuffle_rng_seed: int = 43
    shuffle: bool = True
    shuffle_buffer_size: int = 1024

    warn_on_starvation: bool = True

    rand_augmentation: RandAugmentationConfig = RandAugmentationConfig()

    # This is a hacky workaround that pytorch lightning doesn't seem to allow validation iterator to keep going
    # if you limit the number of validation batches.
    # So we just ignore the iterator
    ignore_iterator_reset: bool = False

    batch_size: int = 32
    dataset_worker_prefetch_factor: int = 2
    dataset_worker_num_workers: int = 1
    shuffled_chunks_queue_size: int = 16

    # For IPC we need a unique id for the dataset.
    # when using multi-GPU the unique id for the same dataset must be the same across all workers.
    # This will get modified by appending a randomly generated string to the end.
    dataset_unique_id: str

    load_video_name: Optional[str] = None

    def get_load_video_name(self):
        if self.load_video_name is None:
            return f"{self.frame_height}x{self.frame_width}.mp4"
        else:
            return self.load_video_name


class ProtoParser(ABC):
    """Parse annotations from a proto file and return the annotations for each chunk."""

    def __init__(self, proto_local_path: str, always_labelled: bool, n_frames: int):
        video_annotation = video_annotation_pb2.VideoAnnotation()
        with open(proto_local_path, "rb") as f:
            file_content = f.read()
            try:
                video_annotation.ParseFromString(file_content)
                self.metadata = video_annotation.metadata
                if len(video_annotation.frame_annotations) == 0:
                    logging.debug(
                        f"skipping the annotation from {proto_local_path} because frame annotation is empty"
                    )
                    self.frame_annotations = None
                else:
                    self.frame_annotations = video_annotation.frame_annotations
            except Exception:
                raise ValueError(f"Error parsing {proto_local_path}")

    def video_id(self) -> Any:
        return self.metadata.id

    @abstractmethod
    def annotate_frames(
        self,
        frames: torch.Tensor,
        start_frame: int,
        end_frame: int,
        valid_frame_mask: torch.Tensor,
    ) -> Any:
        pass


def resize_image_for_model(im: torch.Tensor, inp_dim) -> torch.Tensor:
    assert len(im.shape) == 3
    assert im.shape[0] == 3
    if im.shape[1] == inp_dim[0] and im.shape[2] == inp_dim[1]:
        return im
    resized_im = _canonical_resize(im, inp_dim)
    # resized_im = F.resize(im, inp_dim)
    return resized_im


def _get_zeromq_queue_addr(dataset_unique_id: str, queue_name: str) -> str:
    # To avoid race conditions we create the tempdir in every process (its harmless if it already exists).
    zmq_tempdir = f"/tmp/elefant_zmq/zmq_{dataset_unique_id}"
    os.makedirs(zmq_tempdir, exist_ok=True)

    addr = f"ipc://{zmq_tempdir}/{queue_name}"
    return addr


def _proto_queue_process(cfg, protos, rng, n_preprocess_workers):
    """Queue local proto paths to preprocess workers."""
    logging.info(f"Proto queue process started for dataset {cfg.dataset_unique_id}")

    # Create a queue for each preprocess worker
    proto_queues = [
        zmq_queue.ZMQQueueServer(
            url=_get_zeromq_queue_addr(cfg.dataset_unique_id, f"proto_{i}"),
            per_client_max_size=16,
            n_clients=1,
        )
        for i in range(n_preprocess_workers)
    ]

    epoch = 0
    while True:
        if cfg.shuffle:
            rng.shuffle(protos)

        idx = 0
        for proto in protos:
            success = False
            while not success:
                success = proto_queues[idx].put(proto, wait_seconds=1)
                idx = (idx + 1) % n_preprocess_workers

        # Mark the end of the epoch.
        for i in range(n_preprocess_workers):
            proto_queues[i].put(None, ignore_full=True)

        logging.info(
            f"Added {len(protos)} protos to queue, epoch {epoch} queued for {cfg.dataset_unique_id}"
        )
        epoch += 1


def _preprocess_example(
    proto: str,
    video: str,
    preprocessed_chunks_queue: zmq_queue.ZMQQueueServer,
    proto_parser_factory,
    config,
    rng: np.random.RandomState,
):
    """Parse a proto and video into chunks."""
    decoder = VideoDecoder(video, device="cpu", num_ffmpeg_threads=1)
    n_frames = len(decoder)
    proto_parser = proto_parser_factory(
        proto_local_path=proto,
        always_labelled=config.always_labelled,
        n_frames=n_frames,
    )
    # Check that we have at least T+1 frames so that every chunk has a frame to look ahead.
    if n_frames < 2:
        logging.warning(f"Video {proto} has less than 2 frames, skipping.")
        return

    if config.shuffle:
        # to avoid negative ranges when n_frames < T+1
        max_start = max(0, n_frames - (config.T + 1))
        cap = min(max_start, config.T - 1)
        start_frame0 = rng.randint(0, cap + 1)
    else:
        start_frame0 = 0

    stride = config.T
    start_frame = start_frame0

    # main frames
    while start_frame + (config.T + 1) <= n_frames:
        end_idx = start_frame + (config.T + 1)
        frame_chunk = decoder[start_frame:end_idx]

        if frame_chunk.shape[-2:] != (config.frame_height, config.frame_width):
            assert config.load_video_name == "video.mp4", (
                "The only case we should be resizing is for video.mp4"
            )
            frame_chunk = torch.stack(
                [
                    resize_image_for_model(f, (config.frame_height, config.frame_width))
                    for f in frame_chunk
                ]
            )
        aligned_frames = frame_chunk[:-1]
        valid_mask = torch.ones(config.T, dtype=torch.bool)

        example = proto_parser.annotate_frames(
            aligned_frames,
            start_frame + 1,
            end_idx,
            valid_frame_mask=valid_mask,
        )
        if example is not None:
            preprocessed_chunks_queue.put(example)

        start_frame += stride

    # final shorter tail
    if start_frame < n_frames - 1:
        valid_len = n_frames - 1 - start_frame
        frame_chunk = decoder[start_frame:n_frames]

        if frame_chunk.shape[-2:] != (config.frame_height, config.frame_width):
            assert config.load_video_name == "video.mp4"
            frame_chunk = torch.stack(
                [
                    resize_image_for_model(f, (config.frame_height, config.frame_width))
                    for f in frame_chunk
                ]
            )

        pad_len = config.T - valid_len
        if pad_len > 0:
            last_frame = frame_chunk[-1]
            pad_frames = last_frame.unsqueeze(0).repeat(pad_len, 1, 1, 1)
            frame_chunk = torch.cat([frame_chunk, pad_frames], dim=0)

        aligned_frames = frame_chunk[:-1]
        valid_mask = torch.zeros(config.T, dtype=torch.bool)
        valid_mask[:valid_len] = True

        example = proto_parser.annotate_frames(
            aligned_frames,
            start_frame + 1,
            n_frames,
            valid_frame_mask=valid_mask,
        )
        if example is not None:
            preprocessed_chunks_queue.put(example)
    del decoder


def _preprocess_worker(
    thread_idx: int,
    config,
    proto_parser_factory,
    rng: np.random.RandomState,
):
    preprocessed_chunks_queue = zmq_queue.ZMQQueueServer(
        url=_get_zeromq_queue_addr(
            config.dataset_unique_id, f"preprocess_{thread_idx}"
        ),
        per_client_max_size=config.preprocessed_chunks_queue_size,
        # Only a single shuffle worker.
        n_clients=1,
    )

    proto_queue = zmq_queue.ZMQQueueClient(
        url=_get_zeromq_queue_addr(
            config.dataset_unique_id,
            f"proto_{thread_idx}",
        ),
        client_id=0,
        get_timeout_seconds=60 * 60,
    )
    epoch = 0

    while True:
        try:
            if config.warn_on_starvation:
                queue_wait_start_time = time.time()
            proto_path = proto_queue.get()
            if config.warn_on_starvation:
                queue_wait_time = time.time() - queue_wait_start_time
                if queue_wait_time > 0.05:
                    logging.warning(
                        f"Proto queue {config.dataset_unique_id}/{thread_idx} starved for {queue_wait_time} seconds"
                    )
        except Exception as e:
            logging.error(f"Preprocess worker {thread_idx} failed getting item: {e}")
            time.sleep(0.1)
            continue

        if proto_path is None:
            logging.info(
                f"Proto queue empty, preprocess thread {config.dataset_unique_id}/{thread_idx} finished epoch {epoch}."
            )
            epoch += 1
            preprocessed_chunks_queue.put(None)
            continue

        # Derive video path from proto path
        video_path = os.path.join(
            os.path.dirname(proto_path), config.get_load_video_name()
        )

        if not os.path.exists(proto_path):
            logging.warning(f"Proto file not found: {proto_path}, skipping")
            continue
        if not os.path.exists(video_path):
            logging.warning(f"Video file not found: {video_path}, skipping")
            continue

        try:
            _preprocess_example(
                proto=proto_path,
                video=video_path,
                preprocessed_chunks_queue=preprocessed_chunks_queue,
                proto_parser_factory=proto_parser_factory,
                config=config,
                rng=rng,
            )
        except Exception as e:
            logging.error(
                f"Error preprocess worker {thread_idx} on {proto_path}: {e}",
                exc_info=True,
            )


class VideoProtoDataset(torch.utils.data.IterableDataset, ABC):
    """Dataset of videos and protobufs.

    Intended to be inherited from and the _proto_parser_factory method overridden.
    """

    def __init__(
        self,
        config: VideoProtoDatasetConfig,
        device: str = "cpu",
    ):
        if VideoDecoder is None:
            raise ImportError(
                "VideoDecoder is not installed, `from torchcodec.decoders import VideoDecoder` must have failed."
            )

        self.config = config
        self.device = device

        self._main_thread_init()

    def _proto_parser_factory(
        self, proto_local_path: str, always_labelled: bool, n_frames: int
    ) -> ProtoParser:
        return ProtoParser(proto_local_path, always_labelled, n_frames)

    def _main_thread_init(self):
        """Initialize the dataset on startup, this runs in the main thread."""

        self._epoch = 0
        self._dataset_worker_generation = 0

        self._shuffle_rng = np.random.RandomState(self.config.shuffle_rng_seed)

        # Make sure the folder exists for the zeromq sockets.
        os.makedirs("/tmp/elefant_zmq", exist_ok=True)

        print("LOCAL PREFIX IS ", self.config.local_prefix)

        start_time = time.time()
        self._protos = sorted(
            [
                os.path.join(root, f)
                for root, _, files in os.walk(self.config.local_prefix)
                for f in files
                if f.endswith(".proto")
            ]
        )
        end_time = time.time()
        logging.info(
            f"Time taken to fetch proto files: {end_time - start_time} seconds"
        )

        self._n_protos = len(self._protos)
        if self._n_protos == 0:
            raise ValueError("No protos found in dataset.")

        self._dataset_hash = hash(tuple(self._protos))
        logging.info(f"Dataset hash: {self._dataset_hash}")

        logging.info(f"Found {len(self._protos)} protos in dataset")

        # Shuffle the protos once deterministically.
        np.random.RandomState(0).shuffle(self._protos)

        self._started_subprocesses = False

    def get_n_videos(self) -> int:
        """Number of videos found in the dataset."""
        return self._n_protos

    def get_dataset_hash(self) -> int:
        """Hash of the sorted proto list for reproducibility."""
        return self._dataset_hash

    def _start_queues_and_workers(self):
        """Reset all queues and workers for a new epoch"""
        assert not self._started_subprocesses

        logging.info(f"Starting queues and workers for {self.config.dataset_unique_id}")

        # Make tempdir for the zmq sockets.
        self._zmq_tempdir = f"/tmp/elefant_zmq/zmq_{self.config.dataset_unique_id}"
        os.makedirs(self._zmq_tempdir, exist_ok=True)
        logging.info(f"Using temp zmq dir: {self._zmq_tempdir}")

        n_preprocess_workers = (
            self.config.n_preprocess_workers_per_iter_worker * self._n_dataset_workers
        )

        # Start the proto queue process
        self._proto_queue_process = mp.Process(
            target=_proto_queue_process,
            kwargs={
                "cfg": self.config,
                "protos": self._protos,
                "rng": np.random.RandomState(self._shuffle_rng.randint(0, 1000000)),
                "n_preprocess_workers": n_preprocess_workers,
            },
            daemon=True,
        )
        self._proto_queue_process.start()

        # Start preprocess workers
        self._preprocess_processes = []
        for i in range(n_preprocess_workers):
            process = mp.Process(
                target=_preprocess_worker,
                kwargs={
                    "thread_idx": i,
                    "config": self.config,
                    "proto_parser_factory": self._proto_parser_factory,
                    "rng": np.random.RandomState(self._shuffle_rng.randint(0, 1000000)),
                },
                daemon=True,
            )
            self._preprocess_processes.append(process)
            process.start()

        self._shuffle_thread = elefant_rust.video_proto_dataset.ShuffleThread(
            dataset_unique_id=self.config.dataset_unique_id,
            shuffle=self.config.shuffle,
            shuffle_buffer_size=self.config.shuffle_buffer_size,
            shuffled_chunks_queue_size=self.config.shuffled_chunks_queue_size,
            n_preprocess_workers=n_preprocess_workers,
            n_dataset_workers=self._n_dataset_workers,
            warn_on_starvation=self.config.warn_on_starvation,
            shuffle_rng_seed=self._shuffle_rng.randint(0, 1000000),
        )

        self._started_subprocesses = True

    def _get_shuffle_queue_client(self):
        logging.info(
            f"Worker {self._worker_id} getting shuffle queue client, epoch={self._epoch}"
        )
        shuffle_queue = zmq_queue.ZMQQueueClient(
            url=_get_zeromq_queue_addr(
                self.config.dataset_unique_id,
                f"shuffle",
            ),
            client_id=self._worker_id,
            get_timeout_seconds=60 * 29,
        )
        return shuffle_queue

    def resolved_dataset_unique_id(self) -> Optional[str]:
        """The unique id actually used by the running worker processes"""
        return getattr(self, "_resolved_dataset_unique_id", None)

    def __iter__(self):
        logging.info(
            f"Worker {self._worker_id}/{self.config.dataset_unique_id} starting epoch={self._epoch + 1}, started_subprocesses={self._started_subprocesses}, ignore_iterator_reset={self.config.ignore_iterator_reset}"
        )

        if self._worker_id == 0 and not self._started_subprocesses:
            self._start_queues_and_workers()
            n_examples_in_epoch = 0
        else:
            n_examples_in_epoch = 0
        self._epoch += 1

        if not hasattr(self, "_iter_shuffle_queue"):
            self._iter_shuffle_queue = self._get_shuffle_queue_client()

        while True:
            if self.config.warn_on_starvation:
                queue_wait_start_time = time.time_ns()

            example = self._iter_shuffle_queue.get()

            if self.config.warn_on_starvation:
                queue_wait_time = (time.time_ns() - queue_wait_start_time) / 1e6
                logging.warning(
                    f"Shuffle queue starved for {queue_wait_time} ms, worker_id={self._worker_id}"
                )

            if example is None:
                logging.info(
                    f"{self.config.dataset_unique_id}/{self._worker_id} finished epoch {self._epoch}, config.ignore_iterator_reset={self.config.ignore_iterator_reset}, n_examples_in_epoch={n_examples_in_epoch}"
                )
                if not self.config.ignore_iterator_reset:
                    if n_examples_in_epoch < self.config.batch_size:
                        raise RuntimeError(
                            f"Worker {self.config.dataset_unique_id}/{self._worker_id} epoch {self._epoch} finished, but n_examples_in_epoch={n_examples_in_epoch} < batch_size={self.config.batch_size}"
                        )
                    return
                else:
                    if n_examples_in_epoch == 0:
                        logging.error(
                            f"Worker {self.config.dataset_unique_id}/{self._worker_id} epoch {self._epoch} finished, but had 0 examples."
                        )
                    self._epoch += 1
                    n_examples_in_epoch = 0
            else:
                n_examples_in_epoch += 1
                yield example

    def to_dataloader(self):
        """Using this dataset requires the dataloader to be configured in a specific way, so we provide this as a helper function."""

        queue_server = None

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            if rank == 0:
                dataset_unique_run_id = str(uuid.uuid4())[0:16]
            else:
                dataset_unique_run_id = None
            obj_list = [dataset_unique_run_id]
            torch.distributed.broadcast_object_list(obj_list, 0)
            dataset_unique_run_id = obj_list[0]
        else:
            world_size = 1
            rank = 0
            dataset_unique_run_id = str(uuid.uuid4())[0:16]

        self._dataset_unique_run_id = dataset_unique_run_id
        self._resolved_dataset_unique_id = f"{self.config.dataset_unique_id}_{dataset_unique_run_id}_{self._dataset_worker_generation}"

        def _non_daemonic_worker_init_fn(worker_id):
            dataset_worker_generation = self._dataset_worker_generation

            process = mp.current_process()
            process.daemon = False
            worker_info = torch.utils.data.get_worker_info()

            worker_id_on_this_gpu = worker_info.id
            n_workers_per_gpu = worker_info.num_workers

            global_worker_id = worker_id_on_this_gpu * world_size + rank

            worker_info.dataset._worker_id = global_worker_id
            worker_info.dataset._n_dataset_workers = n_workers_per_gpu * world_size
            worker_info.dataset.config.dataset_unique_id = (
                worker_info.dataset.config.dataset_unique_id
                + f"_{dataset_unique_run_id}_{dataset_worker_generation}"
            )

            logging.info(
                f"Worker {worker_info.dataset._worker_id} starting, n_workers_per_gpu={n_workers_per_gpu}, world_size={world_size}, rank={rank}, "
                + f"dataset_unique_id={worker_info.dataset.config.dataset_unique_id}"
            )

        dataloader = DataLoader(
            self,
            batch_size=self.config.batch_size,
            prefetch_factor=self.config.dataset_worker_prefetch_factor,
            num_workers=self.config.dataset_worker_num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            timeout=60 * 30,
            persistent_workers=True,
            worker_init_fn=_non_daemonic_worker_init_fn,
            in_order=False,
        )

        dataloader._queue_server = queue_server
        return dataloader
