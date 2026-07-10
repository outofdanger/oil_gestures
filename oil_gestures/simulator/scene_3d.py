import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt

from oil_gestures.simulator.render_profile import RenderProfile
from oil_gestures.simulator.render_scheduler import RenderScheduler


class Scene3D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(7, 7, 7, 7)

        # Кросс-платформенный профиль рендера (macOS/Linux/Windows, software-GL,
        # сглаживание, флаг точек-сфер частиц). Управляется env OIL_AA /
        # OIL_POINT_SPHERES / OIL_RENDER_SCALE и т.д.
        self.profile = RenderProfile.detect()
        pv.global_theme.multi_samples = self.profile.multi_samples

        # auto_update=False — забираем рендер себе у pyvistaqt: иначе он держит
        # фоновый таймер (~5 FPS), рендеря вхолостую. Кадрами управляет
        # RenderScheduler: render-on-demand (request_render) + кадровый таймер
        # анимации (start_animation). В покое рендер не идёт вовсе.
        self.plotter = QtInteractor(self, auto_update=False)
        layout.addWidget(self.plotter.interactor)

        self.plotter.interactor.SetInteractorStyle(None)
        self.plotter.interactor.RemoveAllObservers()
        self.plotter.interactor.setMouseTracking(True)
        self.plotter.interactor.setFocusPolicy(Qt.StrongFocus)
        self.plotter.interactor.setFocus()

        ground = pv.Plane(center=(0, -0.02, 0), direction=(0, 1, 0), i_size=18, j_size=24)
        self._ground_actor = self.plotter.add_mesh(ground, color="black", show_edges=False, opacity=0.6)
        self._ground_actor.SetPickable(False)

        self.plotter.background_color = "white"
        self.plotter.view_xy()
        self.plotter.camera.zoom(2.6)
        pos, focus, viewup = self.plotter.camera_position
        new_pos = (pos[0], pos[1] + 5, pos[2])
        new_focus = (focus[0], focus[1] + 5, focus[2])
        self.plotter.camera_position = [new_pos, new_focus, (0, 1, 0)]

        # Единый владелец кадра (render-on-demand + кадровый таймер анимации).
        # Колбэк тика регистрирует Controller через scheduler.set_tick().
        self.scheduler = RenderScheduler(self.plotter, fps=self.profile.animation_fps)

        # Применяем сглаживание/частоту обновления; финальная (де)градация под
        # software-GL — в showEvent, когда GL-контекст уже создан.
        self.profile = self.profile.apply(self.plotter)
        self._profile_finalized = False

    def showEvent(self, event):
        super().showEvent(event)
        # Забираем клавиатурный фокус на 3D-виджет уже ПОСЛЕ показа окна:
        # setFocus() в __init__ (до show) на Windows не прилипает, из-за чего
        # клавиши 1-6/стрелки не доходили до InputHandler.
        self.plotter.interactor.setFocus()
        # На первом показе GL-контекст валиден: перечитываем профиль (ловим
        # software-GL), синхронизируем FPS и рисуем первый кадр (auto_update
        # выключен, поэтому без явного запроса сцена не отрисуется).
        if not self._profile_finalized:
            self._profile_finalized = True
            self.profile = self.profile.apply(self.plotter)
            self.scheduler.set_fps(self.profile.animation_fps)
            self.scheduler.request_render()
            self._log_gl_renderer()

    def _log_gl_renderer(self):
        """Печатает GL_VENDOR/GL_RENDERER — чтобы видеть, на чём реально идёт
        рендер (железный GPU vs software: llvmpipe / GDI Generic / Microsoft)."""
        try:
            caps = self.plotter.render_window.ReportCapabilities() or ""
            for line in caps.splitlines():
                low = line.lower()
                if "opengl vendor" in low or "opengl renderer" in low or "opengl version" in low:
                    print("GL:", line.strip())
        except Exception:
            pass

    def request_render(self):
        """Запросить одиночную перерисовку (коалесцируется планировщиком)."""
        self.scheduler.request_render()

    def update(self):
        # Совместимость: контроллер коллег зовёт scene.update() после разовых
        # изменений сцены — маршрутизируем в render-on-demand.
        self.scheduler.request_render()
