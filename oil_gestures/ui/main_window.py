from PySide6.QtWidgets import QMainWindow, QHBoxLayout, QWidget
from PySide6.QtGui import QIcon

from oil_gestures.simulator.model import Model
from oil_gestures.simulator.camera import Camera
from oil_gestures.simulator.scene_3d import Scene3D
from oil_gestures.simulator.simulator_controller import SimulatorController
from oil_gestures.ui.control_panel import ControlPanel
from oil_gestures.ui.input_handler import InputHandler
from oil_gestures.ui.controller import Controller


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(" Тренажер оператора скважины")
        self.setWindowIcon(QIcon("assets/icon.png"))
        self.resize(1300, 700)

        central = QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet("background-color: gray;")

        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.scene = Scene3D()
        layout.addWidget(self.scene, 8)

        camera = Camera(self.scene.plotter)
        model = Model(self.scene.plotter, "assets/model.glb")

        self.panel = ControlPanel()
        layout.addWidget(self.panel, 2)


        simulator_controller = SimulatorController(model)
        self.controller = Controller(camera, model, self.scene, self.panel, simulator_controller)
        self.input_handler = InputHandler(self.scene.plotter, self.controller)

        print("Тренажер готов к работе!")