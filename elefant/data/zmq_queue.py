import logging
import argparse
import pickle
from elefant_rust import zmq_queue as rust_zmq_queue
import torch
from torch import multiprocessing as mp
import pickle
import time
from multiprocessing.reduction import ForkingPickler
import io
from typing import Optional


SERVER_CLOSE_MSG = b"server_close"


def _set_mp_authkey():
    # For the share memory all the processes need to have the same authkey.
    # We set it here (we rely on firewall for security).
    mp.current_process().authkey = b"elefant"


# We have to set it on import to make sure all processes get it set.
# Even ones that don't use this queue need to match or pytorch will error.
_set_mp_authkey()


class ZMQQueueServer:
    def __init__(self, url: str, per_client_max_size: int, n_clients: int):
        _set_mp_authkey()
        self._url = url
        # logging.info(
        #     f"ZMQQueueServer starting for {url}, max_per_client_size={per_client_max_size}, n_clients={n_clients}"
        # )

        # Use composition instead of inheritance
        self._rust_server = rust_zmq_queue.ZMQQueueServer(
            url, per_client_max_size, n_clients
        )

    def _pickle_item(self, item):
        buf = io.BytesIO()
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(item)
        return buf.getvalue()  # Return bytes, not BytesIO

    def put(
        self, item, wait_seconds: Optional[int] = None, ignore_full: bool = False
    ) -> bool:
        """
        If ignore_full is True, then put will always succeed immediately.
        """
        item = self._pickle_item(item)
        # The Rust implementation now accepts PyBytes directly and releases the GIL
        return self._rust_server.put(item, wait_seconds, ignore_full)

    def put_to_all(self, item):
        logging.info(f"{self._url} ZMQQueueServer putting to all")
        item = self._pickle_item(item)
        # The Rust implementation now accepts PyBytes directly and releases the GIL
        return self._rust_server.put_to_all(item)


class ZMQQueueClient:
    def __init__(
        self, url: str, client_id: int, get_timeout_seconds: Optional[int] = None
    ):
        # logging.info(f"ZMQQueueClient {url}, id {client_id} connecting to server")
        self._url = url
        self._client_id = client_id
        _set_mp_authkey()

        # Use composition instead of inheritance
        self._rust_client = rust_zmq_queue.ZMQQueueClient(
            url, client_id, get_timeout_seconds
        )

    def get(self):
        item = self._rust_client.get()
        if item is None:
            logging.info(
                f"Client {self._url}/{self._client_id} got None (shutdown signal)"
            )
            return None
        # logging.info(
        #     f"Client {self._url}/{self._client_id} got item. item len={len(item)}"
        # )
        item = pickle.loads(item)
        # logging.info(f"Client {self._url}/{self._client_id} unpickled item")
        return item


def _test_client(client: ZMQQueueClient):
    n_items = 0
    while True:
        item = client.get()
        n_items += 1
        logging.info(f"Client got item. n_items={n_items}")
        time.sleep(1)
        if item is None:
            logging.info("Client got None, exiting")
            break


def _test_server(server: ZMQQueueServer):
    i = 0
    while True:
        # Send a large random string.
        server.put((f"test_{i}", torch.randn(1000000)))
        # Send a large random bytes.
        # server.put(torch.randn(1000000).numpy().tobytes())
        i += 1
    logging.info("Server done")


# Used for testing
def _main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", action="store_true")
    ap.add_argument("--client_id", type=int, default=0)
    ap.add_argument("--n_clients", type=int, default=1)
    args = ap.parse_args()

    if args.server:
        server = ZMQQueueServer("ipc:///tmp/zmq_queue_server", 2, args.n_clients)
        _test_server(server)
    else:
        client = ZMQQueueClient(
            "ipc:///tmp/zmq_queue_server", args.client_id, get_timeout_seconds=5
        )
        _test_client(client)


if __name__ == "__main__":
    _main()
