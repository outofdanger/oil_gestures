import vtk
from PySide6.QtCore import QEvent, Qt, QObject


class InputHandler(QObject):
    """
    Только сбор событий и пикинг.
    Всю логику передаёт в контроллер.
    """

    def __init__(self, plotter, controller):
        super().__init__()
        self.plotter = plotter
        self.controller = controller

        self._mouse_pressed = False
        self._mouse_button = None
        self._mouse_press_x = 0
        self._mouse_has_moved = False

        self.plotter.interactor.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseMove:
            return self._on_mouse_move(event)
        elif event.type() == QEvent.MouseButtonPress:
            return self._on_press(event)
        elif event.type() == QEvent.MouseButtonRelease:
            return self._on_release(event)
        elif event.type() == QEvent.Wheel:
            return self._on_wheel(event)
        elif event.type() == QEvent.KeyPress:
            return self._on_key(event)
        return False

    # ========================
    # ПИКИНГ
    # ========================

    def _pick(self, event):
        x = event.position().x()
        y = event.position().y()
        r = self.plotter.renderer
        ws = r.GetSize()
        w = self.plotter.interactor
        vx = int(x * ws[0] / w.width())
        vy = ws[1] - int(y * ws[1] / w.height())
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.Pick(vx, vy, 0, r)
        return picker.GetActor(), picker.GetPickPosition()

    # ========================
    # МЫШЬ
    # ========================

    def _on_mouse_move(self, event):
        actor, _ = self._pick(event)
        self.controller.on_mouse_move(actor)

        if self._mouse_pressed and self._mouse_button == Qt.LeftButton:
            x = event.position().x()
            dx = x - self._mouse_press_x

            if abs(dx) > 1:
                self._mouse_has_moved = True
                self._mouse_press_x = x
                self.controller.on_left_drag(dx)

        return True

    def _on_press(self, event):
        btn = event.button()
        if btn == Qt.LeftButton or btn == Qt.RightButton:
            self._mouse_pressed = True
            self._mouse_button = btn
            self._mouse_press_x = event.position().x()
            self._mouse_has_moved = False
            return True
        return False

    def _on_release(self, event):
        btn = event.button()

        if btn == Qt.LeftButton:
            if self._mouse_pressed and not self._mouse_has_moved:
                actor, position = self._pick(event)
                self.controller.on_left_click(position)
            else:
                self.controller.on_left_release()

            self._mouse_pressed = False
            self._mouse_button = None
            return True

        elif btn == Qt.RightButton:
            if self._mouse_pressed and not self._mouse_has_moved:
                actor, _ = self._pick(event)
                self.controller.on_right_click(actor)

            self._mouse_pressed = False
            self._mouse_button = None
            return True

        return False

    def _on_wheel(self, event):
        delta = event.angleDelta().y()
        self.controller.on_wheel(delta)
        return True

    def _on_key(self, event):
        self.controller.on_key(event.key())
        return True