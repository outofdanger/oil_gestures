from PySide6.QtCore import QObject, QTimer, Qt
from PySide6.QtWidgets import QMenu
from PySide6.QtGui import QCursor
from PySide6.QtCore import Signal

from oil_gestures.integration.contracts import CAMERA_FRAME_CONTRACT
from oil_gestures.simulator.simulator_controller import SimulatorController


class Controller(QObject):
    """
    Дирижёр: вся Qt-специфичная логика приложения (таймер, мышь, клавиатура,
    меню). Интерпретация ML-контракта (жест -> действие) делегирована
    SimulatorController, чтобы эта логика оставалась без зависимости от Qt -
    см. docs/interaction_spec.md и docs/integration_contract.md.
    """
    _ml_event = Signal(dict)

    def __init__(self, camera, model, scene, panel, simulator_controller=None,
                 ml_host="127.0.0.1", ml_port=8765):
        super().__init__()
        self.camera = camera
        self.model = model
        self.scene = scene
        self.panel = panel
        self.simulator_controller = simulator_controller or SimulatorController(model)

        # Таймер
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_tick)
        self._timer_active = False

        # Связи с панелью
        self.panel.emergency_clicked.connect(self._emergency_stop)
        self.panel.inventory_item_clicked.connect(self._on_inventory_click)

        # В __init__:
        self._ml_thread = None
        self._ml_event.connect(self._on_ml_event_main)
        self._start_ml_client(ml_host, ml_port)

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
    #  АВАРИЙНЫЙ СТОП
    # ========================

    def _emergency_stop(self):
        self.model.emergency_stop()
        self.panel.set_message("⚠ АВАРИЙНЫЙ СТОП")


    def _on_inventory_click(self, name: str):
        detail = self.model.get_by_name(name)
        if detail and hasattr(detail, 'attach'):
            detail.attach()
            self.model._active.discard(detail)
            self.panel.set_inventory(self.model.get_inventory())
            self.panel.set_message(f"{name}: установлен(а)")


    def _start_ml_client(self, host="127.0.0.1", port=8765):
        """
        Запустить поток чтения ML-событий через готовый, уже протестированный
        oil_gestures.integration.client.iter_events - вместо ручного парсинга
        NDJSON по сокету. Переподключается с backoff, если ML-сервер ещё не
        поднялся (run_ui.py стартует ML и UI почти одновременно) или обрыв
        соединения произошёл позже.
        """
        import threading
        import time

        from oil_gestures.integration.client import iter_events

        def read_events():
            backoff = 0.5
            while True:
                try:
                    for event in iter_events(host=host, port=port, timeout=5.0):
                        self._ml_event.emit(event)
                    backoff = 0.5  # сервер закрыл поток штатно - не считаем сбоем
                except OSError as exc:
                    print(f"ML client: {exc}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)

        self._ml_thread = threading.Thread(target=read_events, daemon=True)
        self._ml_thread.start()

    def _on_ml_event_main(self, event):
        """Вызывается в главном потоке. Интерпретация - в SimulatorController."""
        if event.get("contract") == CAMERA_FRAME_CONTRACT:
            data = event.get("data_base64")
            if data:
                self.panel.set_camera_frame(data)
            return

        result = self.simulator_controller.handle_event(event)

        self.panel.set_gesture(result.gesture_name, result.gesture_confidence)

        if result.emergency:
            self._emergency_stop()
            return

        if result.message:
            self.panel.set_message(result.message)

        if result.action_taken:
            self._start_timer()