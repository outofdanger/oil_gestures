from PySide6.QtCore import QObject, QPoint, Qt
from PySide6.QtWidgets import QMenu
from PySide6.QtGui import QCursor

from oil_gestures.integration.contracts import CAMERA_FRAME_CONTRACT
from oil_gestures.integration.qt_client import QtEventClient
from oil_gestures.simulator.simulator_controller import SimulatorController
from oil_gestures.ui.camera_frame_decoder import CameraFrameDecoder


class Controller(QObject):
    """
    Дирижёр: вся Qt-специфичная логика приложения (таймер, мышь, клавиатура,
    меню). Интерпретация ML-контракта (жест -> действие) делегирована
    SimulatorController, чтобы эта логика оставалась без зависимости от Qt -
    см. docs/interaction_spec.md и docs/integration_contract.md.
    """

    def __init__(self, camera, model, scene, panel, simulator_controller=None,
                 ml_host="127.0.0.1", ml_port=8765):
        super().__init__()
        self.camera = camera
        self.model = model
        self.scene = scene
        self.panel = panel
        self.simulator_controller = simulator_controller or SimulatorController(model)
        # Текущее открытое контекстное меню (если есть) - чтобы THUMB_UP мог
        # закрыть его программно после выполнения действия (см. on_right_click).
        self._open_menu = None

        # Рендером управляет планировщик сцены (render-on-demand + кадровый
        # таймер анимации). Регистрируем колбэк одного кадра анимации.
        self.scene.scheduler.set_tick(self._frame_tick)

        # Декодирование кадров камеры — в отдельном потоке (latest-wins),
        # чтобы JPEG-декод не конкурировал с рендером 3D-сцены за GUI-поток.
        self._camera_decoder = CameraFrameDecoder(self)
        self._camera_decoder.frame_ready.connect(self.panel.set_camera_image)

        # Связи с панелью
        self.panel.emergency_clicked.connect(self._emergency_stop)
        self.panel.inventory_item_clicked.connect(self._on_inventory_click)

        # Поток событий ML — на нативном QTcpSocket (без Python-потока и без
        # конкуренции за GIL): события приходят прямо в GUI-поток.
        self._ml_client = QtEventClient(ml_host, ml_port, parent=self)
        self._ml_client.event_received.connect(self._on_ml_event_main)
        self._ml_client.start()

    # ========================
    #  КАДР / РЕНДЕР
    # ========================

    def _start_animation(self):
        """Запустить кадровый таймер анимации (камера/вентили/частицы)."""
        self.scene.scheduler.start_animation()

    def _frame_tick(self, dt):
        """Один кадр анимации. Возвращает True, пока что-то ещё движется."""
        self.camera.tick(dt)
        self.model.tick(dt)
        return self.model.has_active() or self.camera.is_rotating()

    # ========================
    #  МЫШЬ — ОТ INPUT HANDLER
    # ========================

    def on_mouse_move(self, actor):
        detail = self.model.get_by_actor(actor)
        if self.model.highlight(detail):
            self.scene.request_render()

    def on_left_drag(self, dx):
        self.camera.start_rotate(-dx * 51)
        self._start_animation()

    def on_left_release(self):
        self.camera.stop_rotate()

    def on_right_click(self):
        detail = self.model.get_highlighted()
        if detail is None:
            return
        actions = self.model.get_menu_actions(detail)
        if not actions:
            return

        # Parented to self.scene (not parent-less) so Qt destroys it in the
        # normal widget-tree order. An unparented popup that outlives the
        # window it was shown over (easy now that popup() below is
        # non-blocking and a gesture-opened menu can sit open indefinitely
        # waiting for THUMB_UP) segfaults in QMenu::hideEvent if the main
        # window closes while it's still open - seen in a real crash dump.
        menu = QMenu(self.scene)
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

        self._open_menu = menu
        menu.aboutToHide.connect(self._on_menu_closed)
        # popup() показывает меню и возвращается немедленно (в отличие от
        # exec(), который крутит вложенный event loop до закрытия меню) -
        # exec() здесь подвешивал рендер 3D-сцены и камеры на всё время,
        # пока меню открыто и ждёт THUMB_UP. С popup() кадровый тик и поток
        # камеры продолжают идти как обычно.
        menu.popup(self._menu_global_pos(detail))

    def _menu_global_pos(self, detail):
        """Где показать меню детали. В жестовом режиме физической мыши на
        объекте нет (она может быть где угодно, даже вне окна), поэтому якорим
        меню к самой детали: проецируем её 3D-центр в экран сцены. Любой сбой
        проекции -> центр виджета сцены; в самом крайнем случае -> позиция
        курсора (прежнее поведение)."""
        interactor = self.scene.plotter.interactor
        try:
            cx, cy, cz = detail.center
            renderer = self.scene.plotter.renderer
            renderer.SetWorldPoint(cx, cy, cz, 1.0)
            renderer.WorldToDisplay()
            dx, dy, _ = renderer.GetDisplayPoint()
            rw, rh = renderer.GetSize()
            if rw > 0 and rh > 0:
                # VTK: origin внизу-слева, физ. пиксели рендера. Qt: вверху-
                # слева, логические пиксели виджета - тот же пересчёт, что в
                # InputHandler._pick_at, но в обратную сторону (учитывает DPR).
                lx = dx * interactor.width() / rw
                ly = (rh - dy) * interactor.height() / rh
                if 0 <= lx <= interactor.width() and 0 <= ly <= interactor.height():
                    return interactor.mapToGlobal(QPoint(int(lx), int(ly)))
            return interactor.mapToGlobal(interactor.rect().center())
        except Exception:
            return QCursor.pos()

    def close_open_menu(self):
        """Force-close a gesture-opened menu before the window goes away -
        belt-and-suspenders alongside parenting it in on_right_click(); see
        MainWindow.closeEvent."""
        if self._open_menu is not None:
            self._open_menu.close()

    def _on_menu_closed(self):
        # aboutToHide срабатывает и при программном close() (THUMB_UP, см.
        # _on_ml_event_main), и при обычном клике вне меню/Esc - в обоих
        # случаях армирование для жестового меню-флоу больше не актуально.
        self._open_menu = None
        self.simulator_controller.clear_armed()

    def _menu_action(self, detail, action):
        self.model.execute_action(detail, action)
        self._start_animation()
        if hasattr(detail, 'state'):
            self.panel.set_inventory(self.model.get_inventory())

    # ========================
    #  КОЛЁСИКО
    # ========================

    def on_wheel(self, delta):
        self.camera.zoom(0.9 if delta > 0 else 1.1)
        self.scene.request_render()

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
                    self.scene.request_render()
                else:
                    ps.start()
                    self._start_animation()
                    self.panel.set_message(f"Частицы {key_str}: вкл")
        elif key == Qt.Key_Up: cam.move(dy=0.3); self.scene.request_render()
        elif key == Qt.Key_Down: cam.move(dy=-0.3); self.scene.request_render()
        elif key == Qt.Key_Left: cam.move(dx=-0.3); self.scene.request_render()
        elif key == Qt.Key_Right: cam.move(dx=0.3); self.scene.request_render()

    # ========================
    #  АВАРИЙНЫЙ СТОП
    # ========================

    def _emergency_stop(self):
        self.model.emergency_stop()
        self.panel.set_message("⚠ АВАРИЙНЫЙ СТОП")
        self.scene.request_render()


    def _on_inventory_click(self, name: str):
        detail = self.model.get_by_name(name)
        if detail and hasattr(detail, 'attach'):
            detail.attach()
            self.model._active.discard(detail)
            self.panel.set_inventory(self.model.get_inventory())
            self.panel.set_message(f"{name}: установлен(а)")
            self.scene.request_render()

    def _on_ml_event_main(self, event):
        """Вызывается в главном потоке. Интерпретация - в SimulatorController."""
        if event.get("contract") == CAMERA_FRAME_CONTRACT:
            data = event.get("data_base64")
            if data:
                self._camera_decoder.submit(data, self.panel.camera_label.size())
            return

        result = self.simulator_controller.handle_event(event)

        self.panel.set_gesture(result.gesture_name, result.gesture_confidence)

        if result.emergency:
            self._emergency_stop()
            return

        if result.message:
            self.panel.set_message(result.message)

        if result.close_menu and self._open_menu is not None:
            self._open_menu.close()

        if result.open_menu and self._open_menu is None:
            self.on_right_click()

        if result.action_taken:
            self._start_animation()