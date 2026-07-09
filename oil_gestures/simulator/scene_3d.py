import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt

from oil_gestures.simulator.render_profile import RenderProfile


class Scene3D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(7, 7, 7, 7)

        # Кросс-платформенный профиль рендера (macOS/Linux/Windows, software-GL,
        # сглаживание, флаг точек-сфер частиц). Управляется env OIL_AA /
        # OIL_POINT_SPHERES / OIL_RENDER_SCALE и т.д. Анимацией/рендером по тику
        # по-прежнему управляет Controller (свой QTimer + scene.update()).
        self.profile = RenderProfile.detect()
        pv.global_theme.multi_samples = self.profile.multi_samples

        self.plotter = QtInteractor(self)
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
        # software-GL) и при необходимости снижаем нагрузку.
        if not self._profile_finalized:
            self._profile_finalized = True
            self.profile = self.profile.apply(self.plotter)

    def update(self):
        self.plotter.update()
