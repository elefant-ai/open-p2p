import abc
import asyncio
import logging
import os
import signal

from elefant.data.proto import video_inference_pb2


UDS_PATH = "/tmp/uds.recap"


class UnixDomainSocketInferenceServer(abc.ABC):
    """Base class for inference servers that communicate over a Unix domain socket."""

    def __init__(self, uds_path: str = UDS_PATH):
        self.uds_path = uds_path
        self.server: asyncio.AbstractServer | None = None
        self.shutdown_event = asyncio.Event()
        self.running = True

    async def _start_server(self) -> None:
        try:
            os.unlink(self.uds_path)
        except OSError:
            if os.path.exists(self.uds_path):
                raise OSError(
                    f"Could not remove existing UDS file {self.uds_path}. Please remove it manually."
                )
        self.server = await asyncio.start_unix_server(
            self._handle_client, self.uds_path, limit=200000
        )
        os.chmod(self.uds_path, 0o777)
        logging.info(f"Server started on {self.uds_path}")

    async def serve(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        await self._start_server()
        await self.shutdown_event.wait()
        await self.shutdown()

    async def shutdown(self) -> None:
        self.running = False
        self.shutdown_event.set()
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        if os.path.exists(self.uds_path):
            try:
                os.unlink(self.uds_path)
            except OSError:
                pass
        logging.info("Server shutdown complete")

    async def _read_frame(
        self, reader: asyncio.StreamReader
    ) -> video_inference_pb2.Frame:
        """Read a frame from the client."""
        frame_length_bytes = await reader.readexactly(4)
        frame_length = int.from_bytes(frame_length_bytes, byteorder="little")
        logging.debug(f"Receiving frame length: {frame_length} bytes")
        frame_data = await reader.readexactly(frame_length)
        logging.debug(f"Received frame with size {len(frame_data)} bytes")
        return video_inference_pb2.Frame.FromString(frame_data)

    async def _write_action(
        self, writer: asyncio.StreamWriter, action: video_inference_pb2.Action
    ) -> None:
        """Write an action to the client."""
        action_data = action.SerializeToString()
        action_length = len(action_data)
        logging.debug(f"Sending action length: {action_length} bytes")
        writer.write(action_length.to_bytes(4, byteorder="little"))
        writer.write(action_data)
        await writer.drain()
        logging.debug("Action sent successfully")

    @abc.abstractmethod
    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Process a single client connection."""
        raise NotImplementedError
