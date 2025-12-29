"""Test script for debugging etc, just to pass random images to the inference server."""

import asyncio
import logging
import time


# import argparse # Remove argparse
import numpy as np

# Assuming the proto generated files are accessible, adjust path if needed
from elefant.data.proto import video_inference_pb2
from elefant.inference.unix_socket_server import UDS_PATH

logging.basicConfig(level=logging.INFO, force=True)

FRAME_WIDTH = 192
FRAME_HEIGHT = 192
FRAME_CHANNELS = 3
FRAME_SLEEP_TIME = 1 / 30


async def frame_generator():
    """Yields random frames indefinitely or up to num_frames."""
    frame_count = 0
    while True:
        # Generate a random HWC frame as bytes
        random_data = np.random.randint(
            0, 256, (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), dtype=np.uint8
        )
        frame_bytes = random_data.tobytes()

        frame_message = video_inference_pb2.Frame(
            data=frame_bytes, width=FRAME_WIDTH, height=FRAME_HEIGHT, id=frame_count
        )
        yield frame_message
        if frame_count % 100 == 0:
            logging.info(f"Sent frame {frame_count}")
        frame_count += 1
        await asyncio.sleep(FRAME_SLEEP_TIME)


async def run_client(uds_path: str = UDS_PATH) -> None:
    """Connect to the server, send frames and log the response FPS."""
    # logging.info(f"Connecting to server at {uds_path}...")
    reader, writer = await asyncio.open_unix_connection(uds_path)

    start_time = time.time()
    frames_processed = 0
    last_log_time = start_time

    frame_processing_times = []

    try:
        async for frame in frame_generator():
            # Send frame
            frame_data = frame.SerializeToString()
            start_send_frame = time.time_ns()
            writer.write(len(frame_data).to_bytes(4, byteorder="little"))
            writer.write(frame_data)
            await writer.drain()

            # Read action
            action_len_bytes = await reader.readexactly(4)
            action_len = int.from_bytes(action_len_bytes, byteorder="little")
            action_data = await reader.readexactly(action_len)
            end_read_action = time.time_ns()
            action = video_inference_pb2.Action.FromString(action_data)

            frame_processing_times.append(end_read_action - start_send_frame)

            frames_processed += 1
            current_time = time.time()
            elapsed_time = current_time - start_time

            if current_time - last_log_time >= 1.0:
                fps = frames_processed / elapsed_time if elapsed_time > 0 else 0
                frame_processing_time_mean = np.mean(frame_processing_times) / 1e6
                frame_processing_time_max = np.max(frame_processing_times) / 1e6
                frame_processing_times = []
                logging.info(
                    f"Received action for frame {action.id}/{frames_processed}: {action.keys}. Current FPS: {fps:.2f}. Frame processing time: {frame_processing_time_mean:.2f} ms (max: {frame_processing_time_max:.2f} ms)"
                )
                last_log_time = current_time

    except asyncio.IncompleteReadError:
        logging.info("Server closed connection")
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception as e:
            logging.error(f"Error closing writer: {e}")

        end_time = time.time()
        total_elapsed = end_time - start_time
        total_fps = frames_processed / total_elapsed if total_elapsed > 0 else 0
        logging.info(f"Total frames processed: {frames_processed}")
        logging.info(f"Total time: {total_elapsed:.2f} seconds")
        logging.info(f"Overall Average FPS: {total_fps:.2f}")


def _main() -> None:
    uds_path = UDS_PATH
    try:
        asyncio.run(run_client(uds_path))
    except KeyboardInterrupt:
        logging.info("Client interrupted by user.")


if __name__ == "__main__":
    _main()
