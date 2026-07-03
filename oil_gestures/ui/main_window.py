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
        self.setWindowIcon(QIcon("assets/ui/icon.png"))
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
        model = Model(
            self.scene.plotter,
            "assets/ui/scene/model.glb",
            particle_count=self.scene.profile.particle_count,
            render_points_as_spheres=self.scene.profile.point_spheres,
        )

        self.panel = ControlPanel()
        layout.addWidget(self.panel, 2)


        simulator_controller = SimulatorController(model)
        self.controller = Controller(camera, model, self.scene, self.panel, simulator_controller)
        self.input_handler = InputHandler(self.scene.plotter, self.controller)

        print("Тренажер готов к работе!")

    def keyPressEvent(self, event):
        # Запасной путь для клавиш: если фокус оказался не на 3D-виджете (и
        # InputHandler.eventFilter их не поймал), нажатие всё равно всплывёт
        # сюда, к главному окну. on_key сам игнорирует неизвестные клавиши, а
        # если 3D-виджет в фокусе - его eventFilter гасит событие раньше, и
        # сюда оно не доходит (двойной обработки нет).
        self.controller.on_key(event.key())
        super().keyPressEvent(event)

    def closeEvent(self, event):
        # A gesture-opened menu (Controller.on_right_click via POINTING_INDEX)
        # can sit open indefinitely waiting for THUMB_UP. Close it explicitly
        # before Qt tears down the window, instead of relying solely on
        # QMenu(self.scene)'s parenting to order destruction correctly -
        # closing the window while it's still open segfaulted in
        # QMenu::hideEvent (seen in a real crash dump) before this existed.
        self.controller.close_open_menu()
        super().closeEvent(event)