"""Декодирование кадров веб-камеры вне GUI-потока.

ML-сервер шлёт кадры как JPEG/base64 (см. oil_gestures.ml.camera_frame). Раньше
base64-декод + JPEG-декод + масштабирование выполнялись прямо в главном потоке
на каждый кадр и конкурировали с рендером 3D-сцены за единственный поток Qt.

Здесь вся тяжёлая работа уходит в отдельный поток, а в главный поток
возвращается уже готовый ``QImage`` (его дёшево превратить в ``QPixmap``).
Политика **latest-wins**: пока поток занят, новые кадры лишь заменяют
«отложенный» — устаревшие кадры дропаются, очередь не растёт.
"""

from __future__ import annotations

from PySide6.QtCore import (
    QByteArray,
    QCoreApplication,
    QMetaObject,
    QMutex,
    QObject,
    QSize,
    Qt,
    QThread,
    Signal,
    Slot,
)
from PySide6.QtGui import QImage


class _DecodeWorker(QObject):
    """Живёт в рабочем потоке. Декодирует только самый свежий кадр."""

    decoded = Signal(QImage)

    def __init__(self) -> None:
        super().__init__()
        self._mutex = QMutex()
        self._pending: tuple[str, QSize] | None = None
        self._busy = False

    def submit(self, payload_base64: str, target: QSize) -> None:
        """Вызывается из GUI-потока. Кладёт кадр как «последний» и будит поток."""
        self._mutex.lock()
        self._pending = (payload_base64, target)
        kick = not self._busy
        if kick:
            self._busy = True
        self._mutex.unlock()
        if kick:
            # Разбудить событийный цикл рабочего потока.
            QMetaObject.invokeMethod(self, "_process", Qt.QueuedConnection)

    @Slot()
    def _process(self) -> None:
        while True:
            self._mutex.lock()
            job = self._pending
            self._pending = None
            if job is None:
                self._busy = False
                self._mutex.unlock()
                return
            self._mutex.unlock()

            payload, target = job
            raw = QByteArray.fromBase64(payload.encode("ascii"))
            image = QImage.fromData(raw, "JPEG")
            if image.isNull():
                continue
            image = image.scaled(
                target,
                Qt.KeepAspectRatio,
                Qt.FastTransformation,  # дёшево; SmoothTransformation тут не нужен
            )
            self.decoded.emit(image)  # доставится в GUI-поток (queued connection)


class CameraFrameDecoder(QObject):
    """Фасад: ``submit(base64, size)`` из GUI-потока → сигнал ``frame_ready(QImage)``."""

    frame_ready = Signal(QImage)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread = QThread()
        self._thread.setObjectName("camera-decode")
        self._worker = _DecodeWorker()
        self._worker.moveToThread(self._thread)
        self._worker.decoded.connect(self.frame_ready)
        self._thread.start()

        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.stop)

    def submit(self, payload_base64: str, target: QSize) -> None:
        self._worker.submit(payload_base64, target)

    def stop(self) -> None:
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)
