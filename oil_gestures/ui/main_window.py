from PySide6.QtWidgets import QMainWindow, QHBoxLayout, QWidget
from PySide6.QtGui import QIcon

from oil_gestures.simulator.model import Model
from oil_gestures.simulator.camera import Camera
from oil_gestures.simulator.scene_3d import Scene3D
from oil_gestures.ui.control_panel import ControlPanel
from oil_gestures.ui.input_handler import InputHandler
from oil_gestures.ui.controller import Controller


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(" Тренажер оператора скважины")
        self.setWindowIcon(QIcon("assets/ui/icon.png"))
        self.resize(1300, 700)
        self.setMinimumSize(1000, 660)

        central = QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet("background-color: gray;")

        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.scene = Scene3D()
        layout.addWidget(self.scene, 8)

        camera = Camera(self.scene.plotter)
        camera.save_default()
        model = Model(
            self.scene.plotter,
            "assets/ui/scene/model.glb",
            render_points_as_spheres=self.scene.profile.point_spheres,
        )

        self.panel = ControlPanel()
        layout.addWidget(self.panel, 2)

        self.controller = Controller(camera, model, self.scene, self.panel)
        self.input_handler = InputHandler(self.scene.plotter, self.controller)

        self.panel.set_inventory(model.get_inventory())
        self.scene.plotter.interactor.setFocus()

        print("Тренажер готов к работе!")

    def keyPressEvent(self, event):
        # Запасной путь для клавиш: если фокус оказался не на 3D-виджете (и
        # InputHandler.eventFilter их не поймал), нажатие всё равно всплывёт
        # сюда, к главному окну. on_key сам игнорирует неизвестные клавиши.
        self.controller.on_key(event.key())
        super().keyPressEvent(event)

    def closeEvent(self, event):
        # Меню, открытое жестом (POINTING_INDEX), может висеть сколько угодно,
        # ожидая THUMB_UP. Закрываем его явно до разрушения окна - иначе
        # безродительский QMenu сегфолтил в QMenu::hideEvent при закрытии окна.
        close_menu = getattr(self.controller, "close_open_menu", None)
        if callable(close_menu):
            close_menu()
        super().closeEvent(event)