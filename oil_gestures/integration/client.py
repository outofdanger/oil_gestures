from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from typing import Any


def iter_events(
    host: str = "127.0.0.1",
    port: int = 8765,
    timeout: float | None = None,
) -> Iterator[dict[str, Any]]:
    """Connect to the ML event stream and yield complete JSON contract events."""

    with socket.create_connection((host, port), timeout=timeout) as connection:
        if timeout is not None:
            connection.settimeout(timeout)
        with connection.makefile("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Truncated tail when the producer disconnects mid-line.
                    break
