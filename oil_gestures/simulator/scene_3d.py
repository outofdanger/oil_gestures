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

        # Кросс-платформенный профиль рендера: всё про macOS/Linux/Windows тут.
        self.profile = RenderProfile.detect()
        # Снимаем дорогой MSAA×8 ещё до создания окна (вместо него — FXAA ниже).
        pv.global_theme.multi_samples = self.profile.multi_samples

        # auto_update=False — забираем рендер себе у pyvistaqt: иначе он держит
        # фоновый таймер на 5 FPS, а на macOS ещё и рендерит в отдельном потоке
        # на каждый кадр. Кадрами теперь управляет RenderScheduler.
        self.plotter = QtInteractor(self, auto_update=False)
        layout.addWidget(self.plotter.interactor)

        self.plotter.interactor.SetInteractorStyle(None)
        self.plotter.interactor.RemoveAllObservers()
        self.plotter.interactor.setMouseTracking(True)
        self.plotter.interactor.setFocusPolicy(Qt.StrongFocus)
        self.plotter.interactor.setFocus()


        ground = pv.Plane(center=(0, -0.02, 0), direction=(0, 1, 0), i_size=12, j_size=20)
        self._ground_actor = self.plotter.add_mesh(ground, color="black", show_edges=False, opacity=0.6)
        self._ground_actor.SetPickable(False)

        self.plotter.background_color = "white"
        self.plotter.view_xy()
        self.plotter.camera.zoom(1.8)
        pos, focus, viewup = self.plotter.camera_position
        new_pos = (pos[0], pos[1] + 4, pos[2])  # выше на 3 единицы
        new_focus = (focus[0], focus[1] + 4, focus[2])
        self.plotter.camera_position = [new_pos, new_focus, (0, 1, 0)]

        # Единый владелец рендера (render-on-demand + кадровый таймер анимации).
        self.scheduler = RenderScheduler(self.plotter, fps=self.profile.animation_fps)

        # Сглаживание/частота обновления; финальная (де)градация — в showEvent,
        # когда GL-контекст уже создан и видно, software это рендер или нет.
        self.profile = self.profile.apply(self.plotter)
        self._profile_finalized = False

    def showEvent(self, event):
        super().showEvent(event)
        # Забираем клавиатурный фокус на 3D-виджет уже ПОСЛЕ показа окна:
        # setFocus() в __init__ (до show) на Windows не прилипает, из-за чего
        # клавиши 1-6/стрелки не доходили до InputHandler (он ловит KeyPress
        # через eventFilter, а тот срабатывает только при наличии фокуса).
        self.plotter.interactor.setFocus()
        # На первом показе GL-контекст уже валиден: перечитываем профиль (так
        # ловится software-GL, напр. llvmpipe на сервере без видеокарты) и при
        # необходимости снижаем нагрузку.
        if not self._profile_finalized:
            self._profile_finalized = True
            self.profile = self.profile.apply(self.plotter)
            self.scheduler.set_fps(self.profile.animation_fps)
            self.scheduler.request_render()

    def request_render(self):
        """Запросить одиночную перерисовку (коалесцируется планировщиком)."""
        self.scheduler.request_render()

    def update(self):
        # Обратная совместимость: старый код мог звать scene.update().
        self.scheduler.request_render()
