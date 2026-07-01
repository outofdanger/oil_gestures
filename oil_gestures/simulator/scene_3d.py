import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt


class Scene3D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(7, 7, 7, 7)

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
        self.plotter.camera.zoom(2.5)
        pos, focus, viewup = self.plotter.camera_position
        new_pos = (pos[0], pos[1] + 4, pos[2])
        new_focus = (focus[0], focus[1] + 4, focus[2])
        self.plotter.camera_position = [new_pos, new_focus, (0, 1, 0)]

    def update(self):
        self.plotter.update()