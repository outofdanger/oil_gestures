"""Асинхронный клиент ML-потока на нативном ``QTcpSocket``.

Раньше события ML читал отдельный Python-поток (``iter_events`` в цикле), и из-за
GIL он конкурировал с главным потоком за исполнение Python-байткода — источник
микро-подёргиваний рендера. Здесь ввод-вывод обслуживает событийный цикл Qt
(C++), Python-потока нет вовсе: ``readyRead`` приходит в поток-владелец (GUI),
парсинг JSON выполняется ровно тогда, когда есть данные.

Формат провода — тот же NDJSON, что у :mod:`oil_gestures.integration.ndjson_server`:
одно JSON-событие на строку, разделитель ``\\n``, без рукопожатия и префикса длины.
Переподключение с экспоненциальным backoff: ML-сервер и UI стартуют почти
одновременно (см. app/run_ui.py), поэтому первые попытки могут не достучаться.
"""

from __future__ import annotations

import json

from PySide6.QtCore import QCoreApplication, QObject, QTimer, Signal
from PySide6.QtNetwork import QAbstractSocket, QTcpSocket


class QtEventClient(QObject):
    """Тонкий клиент NDJSON-потока. Сигналит готовыми словарями событий."""

    event_received = Signal(dict)
    connected = Signal()
    disconnected = Signal()

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        *,
        backoff_ms: int = 500,
        max_backoff_ms: int = 5000,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._host = host
        self._port = int(port)
        self._initial_backoff_ms = int(backoff_ms)
        self._max_backoff_ms = int(max_backoff_ms)
        self._backoff_ms = self._initial_backoff_ms

        self._buffer = bytearray()
        self._closing = False
        self._reconnect_scheduled = False

        self._socket = QTcpSocket(self)
        self._socket.readyRead.connect(self._on_ready_read)
        self._socket.connected.connect(self._on_connected)
        self._socket.disconnected.connect(self._on_disconnected)
        self._socket.errorOccurred.connect(self._on_error)

        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.stop)

    # --------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._closing = False
        self._connect()

    def stop(self) -> None:
        self._closing = True
        # abort() закрывает мгновенно и не шлёт errorOccurred; reconnect подавлен
        # флагом _closing, так что петли переподключения на выходе не будет.
        self._socket.abort()

    def _connect(self) -> None:
        self._reconnect_scheduled = False
        if self._closing:
            return
        self._buffer.clear()
        # Вызывается только из Unconnected-состояния (старт или после
        # disconnected/error), поэтому abort() здесь не нужен.
        self._socket.connectToHost(self._host, self._port)

    def _schedule_reconnect(self) -> None:
        # errorOccurred и disconnected могут прийти на один и тот же сбой —
        # планируем ровно одно переподключение.
        if self._closing or self._reconnect_scheduled:
            return
        self._reconnect_scheduled = True
        delay = self._backoff_ms
        self._backoff_ms = min(self._backoff_ms * 2, self._max_backoff_ms)
        QTimer.singleShot(delay, self._connect)

    # ------------------------------------------------------------ signals
    def _on_connected(self) -> None:
        self._backoff_ms = self._initial_backoff_ms
        # События мелкие и важна задержка — просим у стека низкую задержку
        # (аналог TCP_NODELAY, который сервер ставит на своей стороне).
        self._socket.setSocketOption(QAbstractSocket.SocketOption.LowDelayOption, 1)
        self.connected.emit()

    def _on_disconnected(self) -> None:
        self.disconnected.emit()
        self._schedule_reconnect()

    def _on_error(self, _error: QAbstractSocket.SocketError) -> None:
        self._schedule_reconnect()

    def _on_ready_read(self) -> None:
        self._buffer += bytes(self._socket.readAll())
        # NDJSON: режем по '\n'; неполный хвост остаётся в буфере до след. пакета.
        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                break
            line = bytes(self._buffer[:newline]).strip()
            del self._buffer[: newline + 1]
            if not line:
                continue
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Битая/обрезанная строка — пропускаем, поток продолжается.
                continue
            if isinstance(event, dict):
                self.event_received.emit(event)
