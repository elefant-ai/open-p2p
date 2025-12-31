"""
Echo inference server that sends a blank action in response to each frame.

Only useful for debugging.
Server uses asyncio to handle multiple connections asynchronously.
"""

import logging
import argparse
import asyncio
import os
import time

from elefant.inference.unix_socket_server import (
    UnixDomainSocketInferenceServer,
    UDS_PATH,
)

from elefant.data.proto import video_inference_pb2


class EchoServer(UnixDomainSocketInferenceServer):
    def __init__(self, uds_path: str = UDS_PATH):
        super().__init__(uds_path=uds_path)

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        frame_i = 0
        client_info = writer.get_extra_info("peername")
        logging.info(f"New connection from {client_info}")
        last_frame_read_time = None
        last_action_write_time = None

        try:
            while self.running:
                try:
                    # Read frame and record Unix time
                    frame = await self._read_frame(reader)
                    frame_read_unix_time = time.time()

                    if last_frame_read_time is None:
                        last_frame_read_time = frame_read_unix_time

                    new_frame_read_time = frame_read_unix_time
                    logging.info(
                        f"Frame {frame_i}, Unix time read: {new_frame_read_time:.6f}, time since last frame read: {new_frame_read_time - last_frame_read_time:.6f}s"
                    )
                    last_frame_read_time = new_frame_read_time

                    keys = []
                    mouse_delta_px = None
                    scroll_delta_px = None
                    buttons_down = []
                    frame_i += 1

                    # Create and send response
                    mouse_action = video_inference_pb2.MouseAction()
                    if mouse_delta_px is not None:
                        mouse_action.mouse_delta_px.CopyFrom(mouse_delta_px)
                    if scroll_delta_px is not None:
                        mouse_action.scroll_delta_px.CopyFrom(scroll_delta_px)
                    if buttons_down:
                        mouse_action.buttons_down.extend(buttons_down)

                    # Write action and record Unix time
                    action_write_unix_time = time.time()
                    action = video_inference_pb2.Action(
                        keys=keys,
                        id=frame.id,
                        mouse_action=mouse_action,
                    )
                    await self._write_action(writer, action)

                    if last_action_write_time is None:
                        last_action_write_time = action_write_unix_time

                    new_action_write_time = action_write_unix_time
                    logging.info(
                        f"Action {frame_i - 1}, Unix time written: {new_action_write_time:.6f}, time since last action written: {new_action_write_time - last_action_write_time:.6f}s"
                    )
                    last_action_write_time = new_action_write_time

                except asyncio.IncompleteReadError:
                    logging.info("Client disconnected")
                    break
                except ConnectionError as e:
                    logging.info(f"Connection error: {e}")
                    break
                except Exception as e:
                    logging.error(f"Error processing frame: {e}")
                    break
        finally:
            logging.info("Closing client connection")
            writer.close()
            try:
                await writer.wait_closed()
            except Exception as e:
                logging.error(f"Error while closing writer: {e}")
            logging.info("Client connection closed")


def _main():
    logging.basicConfig(level=logging.INFO)

    # Create server instance
    playback_server = EchoServer()

    try:
        asyncio.run(playback_server.serve())
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        logging.info("Server shutdown complete.")


if __name__ == "__main__":
    _main()
