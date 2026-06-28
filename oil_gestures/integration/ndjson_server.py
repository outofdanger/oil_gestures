from __future__ import annotations

import json
import queue
import socket
import socketserver
import threading
from collections.abc import Mapping
from typing import Any, Protocol


class JsonEvent(Protocol):
    def to_json(self) -> str: ...


class _EventHub:
    def __init__(self, queue_size: int) -> None:
        self.queue_size = queue_size
        self.running = True
        self._clients: set[queue.Queue[bytes]] = set()
        self._lock = threading.Lock()

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def register(self) -> queue.Queue[bytes]:
        client_queue: queue.Queue[bytes] = queue.Queue(maxsize=self.queue_size)
        with self._lock:
            self._clients.add(client_queue)
        return client_queue

    def unregister(self, client_queue: queue.Queue[bytes]) -> None:
        with self._lock:
            self._clients.discard(client_queue)

    def publish(self, payload: bytes) -> None:
        with self._lock:
            clients = tuple(self._clients)
        for client_queue in clients:
            try:
                client_queue.put_nowait(payload)
            except queue.Full:
                try:
                    client_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    client_queue.put_nowait(payload)
                except queue.Full:
                    pass


class _EventHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        hub: _EventHub = self.server.event_hub  # type: ignore[attr-defined]
        client_queue = hub.register()
        try:
            self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while hub.running:
                try:
                    payload = client_queue.get(timeout=0.25)
                except queue.Empty:
                    continue
                self.request.sendall(payload)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            hub.unregister(client_queue)


class _ThreadingEventServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class NDJSONBroadcastServer:
    """Non-blocking, multi-client newline-delimited JSON event broadcaster."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        client_queue_size: int = 8,
    ) -> None:
        if not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if client_queue_size <= 0:
            raise ValueError("client_queue_size must be positive")
        self.host = host
        self.port = port
        self._hub = _EventHub(client_queue_size)
        self._server: _ThreadingEventServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def client_count(self) -> int:
        return self._hub.client_count

    @property
    def address(self) -> tuple[str, int]:
        if self._server is None:
            return self.host, self.port
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> "NDJSONBroadcastServer":
        if self._server is not None:
            return self
        self._hub.running = True
        server = _ThreadingEventServer((self.host, self.port), _EventHandler)
        server.event_hub = self._hub  # type: ignore[attr-defined]
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            name="ml-contract-server",
            daemon=True,
        )
        self._thread.start()
        return self

    def publish(self, event: JsonEvent | Mapping[str, Any]) -> None:
        if self._hub.client_count == 0:
            # No consumer connected: skip serialization entirely.
            return
        if hasattr(event, "to_json"):
            serialized = event.to_json()
        else:
            serialized = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        self._hub.publish((serialized + "\n").encode("utf-8"))

    def close(self) -> None:
        self._hub.running = False
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None

    def __enter__(self) -> "NDJSONBroadcastServer":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
