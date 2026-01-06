# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from typing_extensions import override

from lightning.fabric.plugins import CheckpointIO
from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO

from typing import Union
from pathlib import Path
import logging
import io
import torch


# Modified from:
# From: https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/fabric/plugins/io/torch_io.py#L28
# From: https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/fabric/utilities/cloud_io.py#L73


class ThreadsafeCheckpointer:
    """A checkpointer that saves checkpoints asynchronously in a thread.

    In the main thread it writes the checkpoint to memory, so training can continue.
    In the background thread it writes the checkpoint to disk.

    Args:
        executor: The executor to use for the thread.
    """

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._checkpoint_paths = []

    def save_checkpoint(
        self,
        checkpoint: dict[str, Any],
        path: Union[str, Path],
        storage_options: Optional[Any] = None,
    ) -> None:
        # On the main thread, all we want to do is checkpoint into memory, so training can continue.
        logging.info(f"Saving checkpoint to memory: {path}")
        bytesbuffer = io.BytesIO()
        torch.save(checkpoint, bytesbuffer)
        self._checkpoint_paths.append(str(path))
        self._executor.submit(self._finalize_checkpoint, bytesbuffer, path)

    def _finalize_checkpoint(
        self, bytesbuffer: io.BytesIO, path: Union[str, Path]
    ) -> None:
        logging.info(f"Saving checkpoint to disk: {path}")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bytesbuffer.seek(0)
        with open(path, "wb") as f:
            f.write(bytesbuffer.read())

    def teardown(self) -> None:
        self._executor.shutdown(wait=True)

    def list_checkpoints(self) -> list[str]:
        return self._checkpoint_paths


class AsyncCheckpointIO(_WrappingCheckpointIO):
    """``AsyncCheckpointIO`` enables saving the checkpoints asynchronously in a thread.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Args:
        checkpoint_io: A checkpoint IO plugin that is used as the basis for async checkpointing.

    """

    def __init__(self, checkpoint_io: Optional["CheckpointIO"] = None) -> None:
        super().__init__(checkpoint_io)
        self._checkpointer = None
        self._checkpoint_paths = []

    # CheckpointIO doesn't have a setup method so we have to do something like.
    # We can't do setup in __init__ because if train or validate is called more than once the
    # teardown method deletes the executor.
    def _ensure_setup(self) -> None:
        if self._checkpointer is None:
            self._checkpointer = ThreadsafeCheckpointer()

    @override
    def save_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        """Uses the ``ThreadPoolExecutor`` to save the checkpoints using the base ``checkpoint_io``."""

        self._ensure_setup()

        self._checkpointer.save_checkpoint(*args, **kwargs)

    @override
    def teardown(self) -> None:
        """This method is called to close the threads."""
        if self._checkpointer is not None:
            # Remember the paths we save to.
            self._checkpoint_paths = self._checkpointer.list_checkpoints()
            self._checkpointer.teardown()
            self._checkpointer = None

    def list_checkpoints(self) -> list[str]:
        if self._checkpointer is not None:
            return self._checkpointer.list_checkpoints()
        else:
            return self._checkpoint_paths

    def get_final_checkpoint(self) -> str:
        """Get the final checkpoint (but in a distributed safe way)."""
        if not torch.distributed.is_initialized():
            checkpoints = self.list_checkpoints()
            if len(checkpoints) > 0:
                logging.info(f"Rank 0 final checkpoint: {checkpoints[-1]}")
                return checkpoints[-1]
            else:
                logging.info(f"Rank 0 final checkpoint: None")
                return None
        else:
            # If rank 0
            if torch.distributed.get_rank() == 0:
                checkpoints = self.list_checkpoints()
                if len(checkpoints) > 0:
                    logging.info(f"Rank 0 final checkpoint: {checkpoints[-1]}")
                    final_checkpoint = [checkpoints[-1]]
                else:
                    logging.info(f"Rank 0 final checkpoint: None")
                    final_checkpoint = [None]
            else:
                final_checkpoint = [None]
            torch.distributed.broadcast_object_list(final_checkpoint, 0)
            logging.info(
                f"Rank {torch.distributed.get_rank()} final checkpoint: {final_checkpoint[0]}"
            )
            return final_checkpoint[0]
