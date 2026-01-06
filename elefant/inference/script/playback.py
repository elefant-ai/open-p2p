"""
Playback (open loop) a recap recording.

Only useful for debugging.
Server uses asyncio to handle multiple connections asynchronously.
"""

import logging
import argparse
import asyncio
import os

from elefant.inference.unix_socket_server import (
    UnixDomainSocketInferenceServer,
    UDS_PATH,
)

from elefant.data.proto import video_annotation_pb2
from elefant.data.proto import video_inference_pb2


class PlaybackInferenceServer(UnixDomainSocketInferenceServer):
    def __init__(
        self, annotation: video_annotation_pb2.VideoAnnotation, uds_path: str = UDS_PATH
    ):
        super().__init__(uds_path=uds_path)
        self.annotation = annotation

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        frame_i = 0
        client_info = writer.get_extra_info("peername")
        logging.info(f"New connection from {client_info}")

        try:
            while self.running:
                try:
                    frame = await self._read_frame(reader)

                    # Get user action for this frame
                    if frame_i < len(self.annotation.frame_annotations):
                        user_action = self.annotation.frame_annotations[
                            frame_i
                        ].user_action
                        assert user_action.is_known, (
                            f"User action not known: {user_action}"
                        )  # For now only support human actions.
                        keys = user_action.keyboard.keys
                        mouse_delta_px = user_action.mouse.mouse_delta_px
                        scroll_delta_px = user_action.mouse.scroll_delta_px
                        buttons_down = user_action.mouse.buttons_down
                        logging.info(
                            f"Playing back frame {frame_i} with action {keys}, {mouse_delta_px}"
                        )
                    else:
                        keys = []
                        mouse_delta_px = None
                        scroll_delta_px = None
                        buttons_down = []
                        logging.warning(
                            f"No user action for frame {frame_i}, using empty action."
                        )
                    frame_i += 1

                    # Create and send response
                    mouse_action = video_inference_pb2.MouseAction()
                    if mouse_delta_px is not None:
                        mouse_action.mouse_delta_px.CopyFrom(mouse_delta_px)
                    if scroll_delta_px is not None:
                        mouse_action.scroll_delta_px.CopyFrom(scroll_delta_px)
                    if buttons_down:
                        mouse_action.buttons_down.extend(buttons_down)

                    action = video_inference_pb2.Action(
                        keys=keys,
                        id=frame.id,
                        mouse_action=mouse_action,
                    )

                    await self._write_action(writer, action)

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--proto_path", type=str, required=True)
    args = parser.parse_args()

    proto_path = args.proto_path
    if not proto_path.endswith("annotation.proto"):
        proto_path = os.path.join(proto_path, "annotation.proto")

    logging.info(f"Loading proto from {proto_path}")
    proto = video_annotation_pb2.VideoAnnotation()
    with open(proto_path, "rb") as f:
        proto.ParseFromString(f.read())

    logging.info(f"Found {len(proto.frame_annotations)} frames")

    # Create server instance
    playback_server = PlaybackInferenceServer(proto)

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
