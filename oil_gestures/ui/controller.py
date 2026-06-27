import vtk
from PySide6.QtCore import QObject, QTimer, Qt
from PySide6.QtWidgets import QMenu
from PySide6.QtGui import QCursor


class Controller(QObject):
    """
    Дирижёр: вся логика приложения.
    Знает camera, model, scene, panel.
    Получает события от InputHandler.
    """

    def __init__(self, camera, model, scene, panel):
        super().__init__()
        self.camera = camera
        self.model = model
        self.scene = scene
        self.panel = panel

        # Таймер
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_tick)
        self._timer_active = False

        # Связи с панелью
        self.panel.emergency_clicked.connect(self._emergency_stop)
        self.panel.inventory_item_clicked.connect(self._on_inventory_click)

        # В __init__:
        self._ml_thread = None
        self._start_ml_client()

    # ========================
    #  ТАЙМЕР
    # ========================

    def _start_timer(self):
        if not self._timer_active:
            self._timer.start(16)
            self._timer_active = True

    def _stop_timer(self):
        if not self.model.has_active() and not self.camera.is_rotating():
            self._timer.stop()
            self._timer_active = False

    def _on_tick(self):
        self.camera.tick()
        self.model.tick()
        self.scene.update()
        if not self.model.has_active() and not self.camera.is_rotating():
            self._stop_timer()

    # ========================
    #  МЫШЬ — ОТ INPUT HANDLER
    # ========================

    def on_mouse_move(self, actor):
        detail = self.model.get_by_actor(actor)
        self.model.highlight(detail)

    def on_left_drag(self, dx):
        self.camera.start_rotate(-dx * 51)
        self._start_timer()

    def on_left_release(self):
        self.camera.stop_rotate()

    def on_right_click(self):
        detail = self.model.get_highlighted()
        if detail is None:
            return
        actions = self.model.get_menu_actions(detail)
        if not actions:
            return

        menu = QMenu()
        menu.setStyleSheet("""
            QMenu {
                background-color: white;
                color: black;
                border: 1px solid black;
                padding: 6px;
            }
            QMenu::item {
                padding: 8px 20px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: lightblue;
            }
            QMenu::separator {
                height: 1px;
                background-color: black;
                margin: 4px 10px;
            }
            QMenu::item:disabled {
                color: black;
                background-color: transparent;
            }
        """)
        info = menu.addAction(f"{detail.name} ({type(detail).__name__})")
        info.setEnabled(False)
        menu.addSeparator()

        for label, action in actions:
            menu.addAction(label, lambda a=action, d=detail: self._menu_action(d, a))

        menu.exec(QCursor.pos())

    def _menu_action(self, detail, action):
        self.model.execute_action(detail, action)
        self._start_timer()
        if hasattr(detail, 'state'):
            self.panel.set_inventory(self.model.get_inventory())

    # ========================
    #  КОЛЁСИКО
    # ========================

    def on_wheel(self, delta):
        self.camera.zoom(0.9 if delta > 0 else 1.1)

    # ========================
    #  КЛАВИАТУРА
    # ========================

    def on_key(self, key):
        cam = self.camera

        if key in (Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5, Qt.Key_6):
            key_str = {Qt.Key_1: "1", Qt.Key_2: "2", Qt.Key_3: "3",
                    Qt.Key_4: "4", Qt.Key_5: "5", Qt.Key_6: "6"}[key]
            ps = self.model.particle_systems.get(key_str)
            if ps:
                if ps._active:
                    ps.stop()
                    self.panel.set_message(f"Частицы {key_str}: выкл")
                else:
                    ps.start()
                    self._start_timer()
                    self.panel.set_message(f"Частицы {key_str}: вкл")
        elif key == Qt.Key_Up: cam.move(dy=0.3)
        elif key == Qt.Key_Down: cam.move(dy=-0.3)
        elif key == Qt.Key_Left: cam.move(dx=-0.3)
        elif key == Qt.Key_Right: cam.move(dx=0.3)

    # ========================
    #  ЖЕСТЫ
    # ========================

    def on_gesture(self, name: str, confidence: float = 1.0):
        self.panel.set_message(f"Жест: {name} ({confidence:.0%})")

    # ========================
    #  АВАРИЙНЫЙ СТОП
    # ========================

    def _emergency_stop(self):
        for d in self.model.details:
            if hasattr(d, 'state'):
                d.state = "attached"
                d.show()
            if hasattr(d, 'stop'):
                d.stop()
        self.model._active.clear()
        if self.model.particle_systems:
            for ps in self.model.particle_systems.values():
                ps.stop()
        self.panel.set_message("⚠ АВАРИЙНЫЙ СТОП")
        self.panel.set_inventory([])


    def _on_inventory_click(self, name: str):
        detail = self.model.get_by_name(name)
        if detail and hasattr(detail, 'attach'):
            detail.attach()
            self.model._active.discard(detail)
            self.panel.set_inventory(self.model.get_inventory())
            self.panel.set_message(f"{name}: установлен(а)")


    def _start_ml_client(self):
        """Запустить поток чтения ML-событий."""
        import threading
        import json
        import socket
        
        def read_events():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.connect(("127.0.0.1", 8765))
                buffer = ""
                while True:
                    data = sock.recv(4096).decode()
                    if not data:
                        break
                    buffer += data
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if line:
                            event = json.loads(line)
                            self._on_ml_event(event)
            except Exception as e:
                print(f"ML client: {e}")
            finally:
                sock.close()
        
        self._ml_thread = threading.Thread(target=read_events, daemon=True)
        self._ml_thread.start()

    def _on_ml_event(self, event):
        """Обработать событие от ML-сервера."""
        contract = event.get("contract", "")
        
        if contract == "oil_gestures.ml.camera_frame":
            data = event.get("data_base64", "")
            if data:
                self.panel.set_camera_frame(data)
        
        elif contract == "oil_gestures.ml.runtime":
            gestures = event.get("gestures", {})
            static = gestures.get("static")
            if static:
                self.on_gesture(static)
            cursor = event.get("cursor", {})
            if cursor.get("enabled"):
                self.panel.set_message(f"Курсор: {cursor.get('action', 'NONE')}")