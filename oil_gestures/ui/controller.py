from PySide6.QtCore import QObject, QPoint, QTimer, Qt, Signal
from PySide6.QtWidgets import QMenu
from PySide6.QtGui import QCursor
from oil_gestures.simulator.details_and_particles import Flap, LevelGaugeCover, LevelGaugeScreen, ControllerScreen, ControllerDoor
from PySide6.QtWidgets import QMessageBox

from oil_gestures.integration.contracts import CAMERA_FRAME_CONTRACT
from oil_gestures.simulator.simulator_controller import SimulatorController
from oil_gestures.ui.camera_frame_decoder import CameraFrameDecoder


class Controller(QObject):
    """
    Дирижёр: вся логика приложения.
    Знает camera, model, scene, panel.
    Получает события от InputHandler.
    """
    _ml_event = Signal(dict)

    def __init__(self, camera, model, scene, panel):
        super().__init__()
        self.camera = camera
        self.model = model
        self.scene = scene
        self.panel = panel

        self._timer = QTimer()
        self._timer.timeout.connect(self._on_tick)
        self._timer_active = False

        self._measure_timer = QTimer()
        self._measure_timer.setSingleShot(True)
        self._measure_timer.timeout.connect(self._on_measure_complete)

        self._level_gauge_zoomed = False
        self._controller_zoomed = False
        self._manometer_zoomed = False

        self.panel.help_clicked.connect(self._show_help)
        self.panel.emergency_clicked.connect(self._emergency_stop)
        self.panel.inventory_item_clicked.connect(self._on_inventory_click)

        # Интерпретация ML-контракта (жест -> действие) вынесена в
        # SimulatorController (без зависимости от Qt) - актуальный маппинг из
        # docs/command_mapping.md, включая dynamic-жесты (POINTING_INDEX/SWIPE).
        self.simulator_controller = SimulatorController(model)
        # Текущее открытое жестом меню (POINTING_INDEX), чтобы THUMB_UP мог
        # закрыть его программно после действия.
        self._open_menu = None
        # Декодирование кадров камеры в отдельном потоке (latest-wins).
        self._camera_decoder = CameraFrameDecoder(self)
        self._camera_decoder.frame_ready.connect(self.panel.set_camera_image)

        self._ml_thread = None
        self._ml_event.connect(self._on_ml_event_main)
        self._start_ml_client()

        # Синхронизируем кликабельные зоны экранов с их начальным состоянием
        self._set_level_gauge_regions()
        self._set_controller_regions()

    # ========================
    # ТАЙМЕР
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
    # МЫШЬ — ОТ INPUT HANDLER
    # ========================

    def on_mouse_move(self, actor):
        detail = self.model.get_by_actor(actor)
        self.model.highlight(detail)

    def on_left_drag(self, dx):
        self.camera.start_rotate(-dx * 51)
        self._start_timer()

    def on_left_release(self):
        self.camera.stop_rotate()

    def on_left_click(self, position=None):
        detail = self.model.get_highlighted()
        if detail is None:
            return

        if isinstance(detail, (LevelGaugeScreen, ControllerScreen)):
            if position is not None:
                self._handle_screen_tap(detail, position)
            return

        self._dispatch_click(detail)

    def _handle_screen_tap(self, screen, position):
        target = screen.hit_test(position)
        if not target:
            return

        # Новый случай: выбор пункта меню с визуальным подтверждением
        if isinstance(target, tuple) and len(target) == 3 and target[0] == "select_and_confirm":
            screen_type, index = target[1], target[2]
            if screen_type == "level_gauge":
                ui = self.model.level_gauge_ui
                ui.selected_mode_index = index
                self._refresh_level_gauge_screen()
                # Через 200 мс эмулируем нажатие кнопки подтверждения
                QTimer.singleShot(200, self._confirm_level_gauge_selection)
            elif screen_type == "controller":
                ui = self.model.controller_ui
                ui.selected_mode_index = index
                self._refresh_controller_screen()
                QTimer.singleShot(200, self._confirm_controller_selection)
            return

        # Старая логика для всех остальных случаев (обычные кнопки)
        names = target if isinstance(target, (list, tuple)) else (target,)
        for name in names:
            button = self.model.get_by_name(name)
            if button is not None:
                self._dispatch_click(button)

    def _reset_controller_indicators(self):
        for name in (
                "controller_circle_one",
                "controller_circle_two",
                "controller_circle_three",
                "controller_circle_four",
                "controller_circle_five",
        ):
            indicator = self.model.get_by_name(name)
            if indicator:
                indicator.set_color("silver")

    def _dispatch_click(self, detail):
        if isinstance(detail, Flap):
            self.model.execute_action(detail, "pulse_open")
            self._start_timer()
            self.panel.set_model_state("Flap: стравливание давления")
            return

        if isinstance(detail, LevelGaugeCover):
            if detail._closed:
                self.model.execute_action(detail, "open")
                self.panel.set_model_state("Крышка: открывается")
            else:
                self.model.execute_action(detail, "close")
                self.panel.set_model_state("Крышка: закрывается")
            self._start_timer()
            return

        if isinstance(detail, ControllerDoor):
            if detail._opened:
                self.model.execute_action(detail, "close")
                self.panel.set_model_state("Контроллер: дверца закрывается")
            else:
                self.model.execute_action(detail, "open")
                self.panel.set_model_state("Контроллер: дверца открывается")
            self._start_timer()
            return

        if detail.name == "level_gauge_button_mode":
            self.model.level_gauge_ui.press_mode()
            self._refresh_level_gauge_screen()
            self.scene.update()
            self.panel.set_model_state("Уровнемер: выбор режима")
            return

        if detail.name == "level_gauge_button_input_output":
            ui = self.model.level_gauge_ui
            screen_before = ui.current_screen
            ui.press_input_output()
            if ui.current_screen in ("measure_level", "measure_pressure") and screen_before != ui.current_screen:
                if ui.current_screen == "measure_level":
                    self._try_start_level_measurement()
                else:
                    self._measure_timer.start(2000)
            self._refresh_level_gauge_screen()
            self.scene.update()
            self.panel.set_model_state("Уровнемер: ввод/вывод")
            return

        if detail.name == "level_gauge_button_level":
            self.model.level_gauge_ui.press_level()  # переключает экран на measure_level
            self._try_start_level_measurement()
            self._refresh_level_gauge_screen()
            self.scene.update()
            self.panel.set_model_state("Уровнемер: измерение уровня")
            return

        if detail.name == "level_gauge_button_return":
            self.model.level_gauge_ui.press_return()
            self._refresh_level_gauge_screen()
            self.scene.update()
            self.panel.set_model_state("Уровнемер: главный экран")
            return

        if detail.name == "controller_lever_on_off":
            self.model.controller_ui.toggle_power()
            self.model.execute_action(detail, "toggle")
            self._start_timer()
            self._refresh_controller_screen()

            power = self.model.controller_ui.power_on

            if not power:
                self._reset_controller_indicators()

            power_indicator = self.model.get_by_name("controller_circle_five")
            if power_indicator:
                power_indicator.set_color("green" if power else "silver")

            self.scene.update()
            self.panel.set_model_state(
                "Контроллер: питание включено" if power else "Контроллер: питание выключено"
            )
            return

        if detail.name == "controller_start_button":
            started = self.model.controller_ui.press_start()
            self._refresh_controller_screen()

            if started:
                self._start_timer()

                unwork_indicator = self.model.get_by_name("controller_circle_one")
                work_indicator = self.model.get_by_name("controller_circle_three")
                if work_indicator:
                    work_indicator.set_color("green")
                if unwork_indicator:
                    unwork_indicator.set_color("silver")

                self._flash_detail("controller_circle_two", "yellow", 3000)
                self.panel.set_model_state("Контроллер: пуск")
            else:
                self._flash_detail("controller_circle_five", "red", 1000)
                self.panel.set_model_state("Нет питания")

            self.scene.update()
            return

        if detail.name == "controller_stop_button":
            stopped = self.model.controller_ui.press_stop()
            self._refresh_controller_screen()

            if stopped:
                self._start_timer()

                work_indicator = self.model.get_by_name("controller_circle_three")
                stop_indicator = self.model.get_by_name("controller_circle_one")
                if work_indicator:
                    work_indicator.set_color("silver")
                if stop_indicator:
                    stop_indicator.set_color("red")

                self._flash_detail("controller_circle_two", "yellow", 3000)
                self.panel.set_model_state("Контроллер: стоп")
            else:
                self._flash_detail("controller_circle_five", "red", 1000)
                self.panel.set_model_state("Нет питания")

            self.scene.update()
            return

        if detail.name == "controller_big_button":
            self._perform_controller_emergency_stop("Контроллер: аварийная остановка")
            return

        if detail.name == "controller_button_one":
            ok = self.model.controller_ui.press_menu()
            self._refresh_controller_screen()
            if not ok:
                self._flash_detail("controller_circle_five", "red", 1000)
            self.scene.update()
            self.panel.set_model_state("Контроллер: выбор режима" if ok else "Нет питания")
            return

        if detail.name in {"controller_button_top", "controller_button_right"}:
            ok = self.model.controller_ui.press_next()
            self._refresh_controller_screen()
            if not ok:
                self._flash_detail("controller_circle_five", "red", 1000)
            self.scene.update()
            self.panel.set_model_state("Контроллер: следующий режим" if ok else "Нет питания")
            return

        if detail.name in {"controller_button_lower", "controller_button_left"}:
            ok = self.model.controller_ui.press_prev()
            self._refresh_controller_screen()
            if not ok:
                self._flash_detail("controller_circle_five", "red", 1000)
            self.scene.update()
            self.panel.set_model_state("Контроллер: предыдущий режим" if ok else "Нет питания")
            return

        if detail.name == "controller_button_center":
            ok = self.model.controller_ui.press_confirm()
            self._refresh_controller_screen()
            if ok:
                self._start_timer()
                self._flash_detail("controller_circle_two", "yellow", 600)
            else:
                self._flash_detail("controller_circle_five", "red", 1000)
            self.scene.update()
            self.panel.set_model_state(self.model.controller_ui.status if ok else "Нет питания")
            return

        if detail.name == "controller_button_long":
            ok = self.model.controller_ui.press_back()
            self._refresh_controller_screen()
            if not ok:
                self._flash_detail("controller_circle_five", "red", 1000)
            self.scene.update()
            self.panel.set_model_state("Контроллер: меню" if ok else "Нет питания")
            return

    def _confirm_level_gauge_selection(self):
        """Эмулирует нажатие кнопки 'ВВОД/ВЫВОД' на уровнемере."""
        btn = self.model.get_by_name("level_gauge_button_input_output")
        if btn:
            self._dispatch_click(btn)

    def _confirm_controller_selection(self):
        """Эмулирует нажатие кнопки 'ПОДТВЕРДИТЬ' на контроллере."""
        btn = self.model.get_by_name("controller_button_center")
        if btn:
            self._dispatch_click(btn)

    # ========================
    # ЭКРАНЫ: обновление + кликабельные зоны
    # ========================

    def _refresh_level_gauge_screen(self):
        self.model.update_level_gauge_screen()
        self._set_level_gauge_regions()

    def _refresh_controller_screen(self):
        self.model.update_controller_screen()
        self._set_controller_regions()

    def _set_level_gauge_regions(self):
        screen = self.model.level_gauge_screen
        if screen is None:
            return
        ui = self.model.level_gauge_ui
        regions = {}

        if ui.current_screen == "home":
            # строка "ВЫБОР РЕЖИМА" (см. LevelGaugeUIState.get_lines) = кнопка МЕНЮ
            regions[2] = "level_gauge_button_mode"

        elif ui.current_screen == "mode_select":
            # тап по конкретному пункту меню = столько нажатий "МЕНЮ",
            # сколько нужно долистать до него, плюс одно нажатие
            # "ВВОД/ВЫВОД" для подтверждения — то же самое, что сделал бы
            # оператор физическими кнопками
            n = len(ui.modes)
            for i in range(n):
                regions[2 + i] = ("select_and_confirm", "level_gauge", i)

        elif ui.current_screen == "results":
            # строка с текстом результата = "ВВОД/ВЫВОД" (листает результаты)
            regions[3] = "level_gauge_button_input_output"

        screen.set_regions(regions)

    def _set_controller_regions(self):
        screen = self.model.controller_screen
        if screen is None:
            return
        ui = self.model.controller_ui
        regions = {}

        if ui.power_on and ui.current_screen == "menu":
            n = len(ui.modes)
            for i in range(n):
                regions[1 + i] = ("select_and_confirm", "controller", i)

        elif ui.power_on and ui.current_screen == "data":
            lines = ui.get_lines()
            # Делаем все строки с данными (кроме заголовка) кликабельными
            for idx in range(1, len(lines)):
                regions[idx] = "controller_button_center"

        screen.set_regions(regions)

    def on_right_click(self, actor=None):
        detail = self.model.get_by_actor(actor) if actor is not None else None
        if detail is None:
            return

        self.model.highlight(detail)

        actions = self.model.get_menu_actions(
            detail,
            self._level_gauge_zoomed,
            self._controller_zoomed,
            self._manometer_zoomed
        )
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

        info = menu.addAction(self.model.get_display_name(detail))
        info.setEnabled(False)
        menu.addSeparator()

        status_action = None  # ДИНАМИЧЕСКОЕ ОБНОВЛЕНИЕ: сохраним ссылку на строку статуса

        for label, action in actions:
            if action is None:
                item = menu.addAction(label)
                item.setEnabled(False)
                status_action = item  # ДИНАМИЧЕСКОЕ ОБНОВЛЕНИЕ: запоминаем
            elif action == "partial":
                from PySide6.QtWidgets import QWidget, QVBoxLayout, QSlider, QLabel, QPushButton, QWidgetAction
                slider_menu = QMenu("Выберите % открытия", menu)
                slider_menu.setStyleSheet(menu.styleSheet())

                slider_widget = QWidget()
                slider_layout = QVBoxLayout(slider_widget)
                if hasattr(detail, '_home') and hasattr(detail, '_max') and detail._max > 0:
                    current_percent = int((detail._home / detail._max) * 100)
                else:
                    current_percent = 50

                slider = QSlider(Qt.Horizontal)
                slider.setRange(0, 100)
                slider.setValue(current_percent)
                value_label = QLabel(f"{current_percent}%")
                value_label.setStyleSheet("color: black;")
                slider.valueChanged.connect(lambda v: value_label.setText(f"{v}%"))
                slider_layout.addWidget(value_label)
                slider_layout.addWidget(slider)

                apply_btn = QPushButton("Применить")
                # ДИНАМИЧЕСКОЕ ОБНОВЛЕНИЕ: передаём status_action в замыкание
                apply_btn.clicked.connect(
                    lambda: self._menu_action(detail, f"set_{slider.value()}", status_action)
                )
                slider_layout.addWidget(apply_btn)

                slider_action = QWidgetAction(menu)
                slider_action.setDefaultWidget(slider_widget)
                slider_menu.addAction(slider_action)

                menu.addMenu(slider_menu)
            else:
                # ДИНАМИЧЕСКОЕ ОБНОВЛЕНИЕ: передаём status_action в замыкание
                menu.addAction(
                    label,
                    lambda a=action, d=detail, sa=status_action: self._menu_action(d, a, sa)
                )

        menu.exec(QCursor.pos())

    def _menu_action(self, detail, action, status_action=None):
        if action == "focus_level_gauge":
            assembly = getattr(detail, "parent_assembly", None)
            if assembly and not self._level_gauge_zoomed:
                self.camera.save_current_view()
                self.camera.focus_on_level_gauge(assembly.bounds)
                self._level_gauge_zoomed = True
                self.scene.update()
                self.panel.set_model_state("Уровнемер: приближение")
            return

        if action == "unfocus_level_gauge":
            if self._level_gauge_zoomed:
                self.camera.restore_saved_view()
                self._level_gauge_zoomed = False
                self._manometer_zoomed = False
                self.scene.update()
                self.panel.set_model_state("Уровнемер: отдаление")
            return

        if action == "focus_controller":
            bounds = self.model.get_controller_bounds()
            if bounds is not None and not self._controller_zoomed:
                self.camera.save_current_view()
                self.camera.focus_on_controller(bounds)
                self._controller_zoomed = True
                self.scene.update()
                self.panel.set_model_state("Контроллер: приближение")
            return

        if action == "unfocus_controller":
            if self._controller_zoomed:
                self.camera.restore_saved_view()
                self._controller_zoomed = False
                self.scene.update()
                self.panel.set_model_state("Контроллер: отдаление")
            return

        if action == "focus_manometer":
            if not self._manometer_zoomed:
                self.camera.save_current_view()
                self.camera.focus_on_manometer(detail.bounds)
                self._manometer_zoomed = True
                self.scene.update()
                self.panel.set_model_state("Манометр: приближение")
            return

        if action == "unfocus_manometer":
            if self._manometer_zoomed:
                self.camera.restore_saved_view()
                self._manometer_zoomed = False
                self.scene.update()
                self.panel.set_model_state("Манометр: отдаление")
            return

        # Проверка перед снятием уровнемера
        if action == "remove_level_gauge":
            min_percent = self.model.get_pressure_for_blocker("plug")
            if min_percent > 0:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Опасно!")
                msg.setText(f"Нельзя снять под давлением\nСначала закройте вентили.")
                msg.exec()
                return

        # Проверка перед снятием: нельзя снимать под давлением
        if action == "remove":
            min_percent = self.model.get_pressure_for_blocker(detail.name)
            if min_percent > 0:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Опасно!")
                msg.setText(f"Нельзя снять под давлением\nСначала закройте вентили.")
                msg.exec()
                return

        self.model.execute_action(detail, action)
        if action.startswith("set_") and status_action is not None:
            self._update_status_when_done(detail, status_action)
        self._start_timer()
        self.panel.set_inventory(self.model.get_inventory())

        if action == "remove_level_gauge":
            self._level_gauge_zoomed = False


    def _update_status_action(self, detail, status_action):
        if status_action is None:
            return
        if hasattr(detail, '_home') and hasattr(detail, '_max') and detail._max > 0:
            percent = int((detail._home / detail._max) * 100)
        else:
            percent = 0
        text = "Закрыто" if percent == 0 else f"Открыта: {percent}%"
        status_action.setText(text)

    def _update_status_when_done(self, detail, status_action):
        """Периодически проверяет, завершилась ли анимация, и обновляет статус."""
        if status_action is None:
            return
        if detail.has_animation():
            # Анимация ещё идёт — проверяем через 100 мс
            QTimer.singleShot(100, lambda: self._update_status_when_done(detail, status_action))
        else:
            # Анимация завершена — обновляем статус
            self._update_status_action(detail, status_action)

    # ========================
    # КОЛЁСИКО
    # ========================

    def on_wheel(self, delta):
        self.camera.zoom(0.9 if delta > 0 else 1.1)

    # ========================
    # КЛАВИАТУРА
    # ========================

    def on_key(self, key):
        cam = self.camera

        # Ручной тумблер частиц клавишами 1-6 намеренно убран: поток теперь
        # прод-логика в Model.tick() (нефть/газ идут там, где цепочка вентилей
        # открыта + контроллер запущен + не перекрыто блокером). Ручной запуск
        # тик всё равно перебивал бы на следующем кадре. Открывай вентили
        # (мышью/меню или жестом THUMB_UP) - поток пойдёт сам.
        if key == Qt.Key_Up:
            cam.move(dy=0.3)
        elif key == Qt.Key_Down:
            cam.move(dy=-0.3)
        elif key == Qt.Key_Left:
            cam.move(dx=-0.3)
        elif key == Qt.Key_Right:
            cam.move(dx=0.3)
        elif key == Qt.Key_Escape:
            cam.reset()
            self._level_gauge_zoomed = False
            self._controller_zoomed = False
            self._manometer_zoomed = False
            self.scene.update()
            self.panel.set_model_state("Исходный вид")

    # ========================
    # ЖЕСТЫ
    # ========================

    def on_gesture(self, name: str, confidence: float = 1.0):
        detail = self.model.get_highlighted()

        if name in ("FIST", "Closed_Fist"):
            if detail and hasattr(detail, 'close') and not detail.has_animation():
                self.model.execute_action(detail, 'close')
                self._start_timer()
                self.panel.set_model_state(f"✊ Закрыть: {detail.name}")
            else:
                self.panel.set_model_state("✊ Кулак (нет цели)")

        elif name in ("OPEN_PALM", "Open_Palm"):
            if detail and hasattr(detail, 'open') and not detail.has_animation():
                self.model.execute_action(detail, 'open')
                self._start_timer()
                self.panel.set_model_state(f"✋ Открыть: {detail.name}")
            else:
                self.panel.set_model_state("✋ Ладонь (нет цели)")

        elif name in ("THUMB_UP", "Thumb_Up"):
            self._emergency_stop()
            self.panel.set_model_state("👍 Аварийный стоп")

        elif name in ("VICTORY", "Victory"):
            self.panel.set_model_state("✌ Победа (курсор вкл/выкл)")

        else:
            self.panel.set_model_state(f"Жест: {name} ({confidence:.0%})")

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
                            self._ml_event.emit(event)
            except Exception as e:
                print(f"ML client: {e}")
            finally:
                sock.close()

        self._ml_thread = threading.Thread(target=read_events, daemon=True)
        self._ml_thread.start()

    def _on_ml_event_main(self, event):
        """Вызывается в главном потоке. Интерпретация - в SimulatorController;
        здесь только Qt-побочные эффекты (камера, панель, меню, таймер)."""
        # Кадры камеры - в отдельный декодер (JPEG/base64 -> QImage вне GUI-потока).
        if event.get("contract") == CAMERA_FRAME_CONTRACT:
            data = event.get("data_base64")
            if data:
                self._camera_decoder.submit(data, self.panel.camera_label.size())
            return

        result = self.simulator_controller.handle_event(event)

        if result.gesture_name:
            self.panel.set_gesture(f"{result.gesture_name} ({result.gesture_confidence:.0%})")

        if result.emergency:
            self._emergency_stop()
            return

        if result.message:
            self.panel.set_model_state(result.message)

        if result.close_menu and self._open_menu is not None:
            self._open_menu.close()

        if result.open_menu and self._open_menu is None:
            self._open_gesture_menu()

        if result.action_taken:
            self._start_timer()

    # ========================
    # ЖЕСТОВОЕ МЕНЮ (POINTING_INDEX -> открыть, THUMB_UP -> действие)
    # ========================

    def _open_gesture_menu(self):
        """Неблокирующее меню для выделенной детали (визуальное подтверждение
        перед THUMB_UP). Отдельно от on_right_click мыши: popup вместо exec,
        чтобы не подвешивать рендер, и с родителем, чтобы не сегфолтить при
        закрытии окна."""
        detail = self.model.get_highlighted()
        if detail is None:
            return
        actions = self.model.get_menu_actions(
            detail, self._level_gauge_zoomed, self._controller_zoomed, self._manometer_zoomed
        )
        if not actions:
            return
        menu = QMenu(self.scene)
        info = menu.addAction(self.model.get_display_name(detail))
        info.setEnabled(False)
        menu.addSeparator()
        for label, action in actions:
            if action in (None, "partial"):
                continue
            menu.addAction(label, lambda a=action, d=detail: self._menu_action(d, a))
        self._open_menu = menu
        menu.aboutToHide.connect(self._on_menu_closed)
        menu.popup(self._menu_global_pos(detail))

    def close_open_menu(self):
        """Закрыть жестовое меню до разрушения окна (см. MainWindow.closeEvent)."""
        if self._open_menu is not None:
            self._open_menu.close()

    def _on_menu_closed(self):
        self._open_menu = None
        self.simulator_controller.clear_armed()

    def _menu_global_pos(self, detail):
        """Позиция меню над выделенной деталью (в жестовом режиме мыши на
        объекте нет). Фолбэк: центр виджета сцены, затем позиция курсора."""
        interactor = self.scene.plotter.interactor
        try:
            cx, cy, cz = detail.center
            renderer = self.scene.plotter.renderer
            renderer.SetWorldPoint(cx, cy, cz, 1.0)
            renderer.WorldToDisplay()
            dx, dy, _ = renderer.GetDisplayPoint()
            rw, rh = renderer.GetSize()
            if rw > 0 and rh > 0:
                lx = dx * interactor.width() / rw
                ly = (rh - dy) * interactor.height() / rh
                if 0 <= lx <= interactor.width() and 0 <= ly <= interactor.height():
                    return interactor.mapToGlobal(QPoint(int(lx), int(ly)))
            return interactor.mapToGlobal(interactor.rect().center())
        except Exception:
            return QCursor.pos()

    # ========================
    # ИЗМЕРЕНИЕ
    # ========================

    def _on_measure_complete(self):
        ui = self.model.level_gauge_ui
        # Если ошибки нет и таймер сработал – завершаем измерение
        if not ui._level_measurement_failed:
            if ui.current_screen == "measure_level":
                ui.complete_level_measurement()
            elif ui.current_screen == "measure_pressure":
                ui._pressure_measured = True
        self._refresh_level_gauge_screen()
        self.scene.update()
        self.panel.set_model_state("Уровнемер: измерение завершено")

    def _try_start_level_measurement(self):
        ui = self.model.level_gauge_ui
        valve = self.model.get_by_name("valve_1")
        is_running = self.model.controller_ui.power_on and self.model.controller_ui.running
        valve_open = (valve is not None and valve._home > 0)

        if not valve_open and not is_running:
            ui._level_measurement_failed = True
            ui._level_error_lines = ["Запустите установку и", "откройте затрубную задвижку!"]
            ui._level_measured = False
            ui._last_level_m = None
        elif not valve_open and is_running:
            ui._level_measurement_failed = True
            ui._level_error_lines = ["Откройте затрубную задвижку"]
            ui._level_measured = False
            ui._last_level_m = None
        elif valve_open and not is_running:
            ui._level_measurement_failed = True
            ui._level_error_lines = ["Запустите установку!"]
            ui._level_measured = False
            ui._last_level_m = None
        else:
            ui._level_measurement_failed = False
            ui._level_error_lines = []
            self._measure_timer.start(2000)

        self._refresh_level_gauge_screen()
        self.scene.update()
        if ui._level_measurement_failed:
            self.panel.set_model_state("Уровнемер: ошибка измерения")
        else:
            self.panel.set_model_state("Уровнемер: измерение запущено")
        return not ui._level_measurement_failed
    # ========================
    # АВАРИЙНЫЙ СТОП
    # ========================

    def _perform_controller_emergency_stop(self, message):
        self.model.controller_ui.force_power_off()

        lever = self.model.get_by_name("controller_lever_on_off")
        if lever:
            self.model.execute_action(lever, "force_off")

        if self.model.particle_systems:
            for ps in self.model.particle_systems.values():
                ps.stop()

        self._reset_controller_indicators()
        self._refresh_controller_screen()
        self._start_timer()
        self.scene.update()
        self.panel.set_model_state(message)

    def _emergency_stop(self):
        self._perform_controller_emergency_stop("⚠ АВАРИЙНЫЙ СТОП")
        self._level_gauge_zoomed = False
        self._controller_zoomed = False
        self._manometer_zoomed = False

    def _on_inventory_click(self, name: str):
        # Проверка: установка уровнемера при наличии заглушки
        if name == "level_gauge":
            plug = self.model.get_by_name("plug")
            if plug and plug.state == "attached":
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setWindowTitle("Невозможно установить")
                msg_box.setText("Сначала снимите заглушку!")
                msg_box.setStandardButtons(QMessageBox.Ok)
                msg_box.exec()
                return

        # Проверка: установка заглушки при установленном уровнемере
        if name == "plug":
            level_gauge_assembly = self.model.level_gauge_assembly
            if level_gauge_assembly and level_gauge_assembly.state == "attached":
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setWindowTitle("Невозможно установить")
                msg_box.setText("Сначала снимите уровнемер!")
                msg_box.setStandardButtons(QMessageBox.Ok)
                msg_box.exec()
                return

        # Проверка: нельзя установить если в цепочке давление
        if name == "level_gauge":
            min_percent = self.model.get_pressure_for_blocker("plug")
        else:
            min_percent = self.model.get_pressure_for_blocker(name)
        if min_percent > 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Опасно!")
            msg.setText(f"Нельзя установить под давлением\nСначала закройте вентили.")
            msg.exec()
            return


        # Если все проверки пройдены, выполняем установку
        detail = self.model.get_by_name(name)
        if detail and hasattr(detail, "attach"):
            detail.attach()
            self.model._active.discard(detail)
            self.panel.set_inventory(self.model.get_inventory())
            self.panel.set_model_state(f"{name}: установлен(а)")


    def _show_help(self):
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QScrollArea, QWidget

        dlg = QDialog()
        dlg.setWindowTitle("Инструкция по управлению")
        dlg.setFixedSize(650, 600)
        dlg.setStyleSheet("background-color: white;")

        outer_layout = QVBoxLayout(dlg)
        outer_layout.setContentsMargins(10, 10, 10, 10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: white; border: none;")

        content = QWidget()
        content.setStyleSheet("background-color: white;")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        text = QLabel("""
        🖱️ Мышь:
        • ЛКМ + тянуть — вращение камеры
        • ПКМ — меню действий над деталью
        • Колёсико — zoom

        ⌨️ Клавиши:
        • 1-4 — ракурсы камеры
        • R — сброс камеры
        • Esc — возврат к исходному виду
        • Стрелки — движение камеры
        • 1-6 — вкл/выкл частицы

        ✋ Жесты (когда курсор выключен):
        • ✊ Кулак — закрыть
        • ✋ Ладонь — открыть
        • 👍 Палец вверх — аварийный стоп
        • ✌ Виктори — вкл/выкл курсор
        """)
        text.setStyleSheet("color: black; font-size: 13px;")
        layout.addWidget(text)

        controller_text = QLabel("""
        🎛️ Контроллер — порядок работы:
         1. Открыть дверцу контроллера.
         2. Включить питание рычагом.
           После включения должен загореться индикатор питания.
         3. Использовать кнопки управления:
            МЕНЮ — открыть список режимов.
            СЛЕД / ПРЕД — выбрать режим.
            ОК — подтвердить выбор.
            НАЗАД — вернуться на предыдущий экран.
         4. Режим "Получить данные":
            показывает текущие параметры:
            ток, напряжение, давление, частота, cosφ.
         5. Режим "Повысить дебит":
            увеличивает ток, частоту и заданное давление.
         6. Большая желтая кнопка — аварийная остановка.
            При нажатии контроллер выключается,
            рычаг возвращается вверх,
            индикаторы гаснут.
         7. Для выключения контроллера можно также перевести рычаг вверх.
         
         ⚠️ Важно:
        • Пока установка не запущена — задвижки можно открывать/закрывать, 
          но давления, потока и показаний приборов не будет.
          Сначала включите питание и нажмите ПУСК.
        """)
        controller_text.setStyleSheet("color: black; font-size: 13px;")
        layout.addWidget(controller_text)
        
        level_gauge_text = QLabel("""
        📊 Уровнемер — порядок работы:
         1. Убрать заглушку
         2. Открыть внешнюю затрубную задвижку
         3. Убедиться, что есть давление по манометру
            (манометр насоса покажет давление, только
            если вместе с задвижкой открыт ещё и
            соседний вентиль)
         4. Продуть линию
         5. Закрыть задвижку
         6. Установить уровнемер
         7. Выбрать режим кнопкой РЕЖИМ, подтвердить
            кнопкой ВВОД/ВЫВОД (или кликами по экрану)
         8. Открыть внешнюю затрубную задвижку
         9. Резко ударить по рычагу клапана
         10. На табло появится значение уровня
         11. Закрыть внешнюю затрубную задвижку
         12. Стравить давление через клапан
         13. Снять уровнемер

        ⚠️ Важно:
        • Пока установлена заглушка, поставить уровнемер нельзя,
          и наоборот.
        • Снять или поставить заглушку/уровнемер нельзя
          под давлением — сначала закройте задвижку.
        • Давление в линии уровнемера появится, только
          если установка включена и запущена, задвижка
          открыта и уровнемер уже установлен.
        • Замер уровня запускается кнопками РЕЖИМ +
          ВВОД/ВЫВОД либо кнопкой УРОВЕНЬ — если на
          этот момент задвижка ещё закрыта или установка
          не запущена, экран покажет ошибку, и режим
          нужно будет подтвердить повторно.
        • Просмотреть уже полученные результаты можно,
          выбрав режим "Просмотр результатов".
        """)
        level_gauge_text.setStyleSheet("color: black; font-size: 13px;")
        layout.addWidget(level_gauge_text)

        layout.addStretch()
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

        dlg.exec()

    # А это зона того, что добавил Влад

    def _flash_detail(self, detail_name, color, duration_ms=3000):
        detail = self.model.get_by_name(detail_name)
        if detail is None:
            return

        original_color = detail._original_color
        detail.set_color(color)
        self.scene.update()

        def restore_color():
            detail.actor.GetProperty().SetColor(*original_color)
            detail._original_color = original_color
            self.scene.update()

        QTimer.singleShot(duration_ms, restore_color)