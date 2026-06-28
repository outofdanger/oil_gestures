"""Единый владелец рендера сцены — кросс-платформенно и детерминированно.

Зачем: по умолчанию pyvistaqt держит собственный фоновый таймер
(`auto_update=5.0` → перерисовка 5 раз/сек), а на macOS ещё и оборачивает
`render()` в поток на каждый кадр. Из-за этого на разных ОС поведение разное и
непредсказуемое. Здесь мы:

* выключаем авто-таймер pyvistaqt (`auto_update=False` при создании плоттера —
  это делает Scene3D), и
* рендерим сами, синхронно, прямым вызовом ``render_window.Render()`` из
  GUI-потока. Интерактор pyvistaqt построен на ``QWidget`` (нативное VTK-окно),
  поэтому прямой Render() безопасен на macOS/Linux/Windows и не плодит потоки.

Две модели обновления, как в нормальных 3D-приложениях:

* **render-on-demand** — :meth:`request_render` рисует один кадр, коалесцируя
  множественные запросы в пределах витка цикла событий (навёл мышь, крутанул
  zoom, переставил деталь);
* **animation clock** — :meth:`start_animation` запускает кадровый таймер,
  который каждый кадр зовёт ``tick(dt)`` и рисует, пока tick возвращает True
  (идёт вращение камеры / анимация вентиля / летят частицы). В покое таймер
  остановлен → 0% CPU/GPU вхолостую.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject, Qt, QTimer


class RenderScheduler(QObject):
    """Управляет тем, *когда* рисуется кадр. Не знает про доменную логику —
    логику кадра передаёт зарегистрированный через :meth:`set_tick` колбэк."""

    def __init__(self, plotter, *, fps: int = 60, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._plotter = plotter
        self._render_window = plotter.render_window
        self._tick: Callable[[float], bool] | None = None

        fps = max(1, int(fps))
        self._dt = 1.0 / fps
        self._frame_timer = QTimer(self)
        self._frame_timer.setTimerType(Qt.PreciseTimer)
        self._frame_timer.setInterval(max(1, int(1000.0 / fps)))
        self._frame_timer.timeout.connect(self._on_frame)

        self._animating = False
        self._render_pending = False

    # ------------------------------------------------------------- tick
    def set_fps(self, fps: int) -> None:
        """Сменить частоту кадров анимации (напр., после деградации профиля)."""
        fps = max(1, int(fps))
        self._dt = 1.0 / fps
        self._frame_timer.setInterval(max(1, int(1000.0 / fps)))

    def set_tick(self, tick: Callable[[float], bool]) -> None:
        """Зарегистрировать колбэк кадра анимации.

        ``tick(dt) -> bool``: продвинуть симуляцию на ``dt`` секунд и вернуть
        True, если анимация должна продолжаться (что-то ещё движется)."""
        self._tick = tick

    # --------------------------------------------------- render-on-demand
    def request_render(self) -> None:
        """Запросить одиночную перерисовку (коалесцируется в один кадр).

        Во время анимации игнорируется — кадровый таймер и так рисует."""
        if self._animating or self._render_pending:
            return
        self._render_pending = True
        # singleShot(0) откладывает рендер до конца текущего витка цикла
        # событий, поэтому несколько request_render() подряд дают один кадр.
        QTimer.singleShot(0, self._flush_render)

    def _flush_render(self) -> None:
        self._render_pending = False
        if not self._animating:
            self._render()

    # ------------------------------------------------------ animation clock
    def start_animation(self) -> None:
        """Запустить кадровый таймер (если ещё не запущен)."""
        if not self._animating:
            self._animating = True
            self._render_pending = False
            self._frame_timer.start()

    def stop_animation(self) -> None:
        """Принудительно остановить кадровый таймер (напр., аварийный стоп)."""
        if self._animating:
            self._animating = False
            self._frame_timer.stop()

    @property
    def is_animating(self) -> bool:
        return self._animating

    def _on_frame(self) -> None:
        keep_going = bool(self._tick(self._dt)) if self._tick is not None else False
        self._render()
        if not keep_going:
            self.stop_animation()

    # ------------------------------------------------------------- render
    def _render(self) -> None:
        try:
            self._render_window.Render()
        except RuntimeError:
            # Окно уничтожено при закрытии приложения — гасим гонку на выходе.
            self.stop_animation()
