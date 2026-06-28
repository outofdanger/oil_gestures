import platform

import vtk
from PySide6.QtCore import QEvent, Qt, QObject, QTimer
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication


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

        # Один пикер на всё время жизни: создавать vtkCellPicker на каждое
        # движение мыши дорого и не даёт VTK ничего кэшировать.
        self._picker = vtk.vtkCellPicker()
        self._picker.SetTolerance(0.005)

        self.plotter.interactor.installEventFilter(self)

        # macOS: виджет VTK создаётся как нативное окно (rwi ставит
        # WA_PaintOnScreen), и Qt не доставляет hover-события (mouse move без
        # зажатой кнопки) на нативный NSView — из-за этого подсветка детали
        # включалась только при зажатой ЛКМ. На Linux/Windows событийный путь
        # через eventFilter работает, поэтому там опрос не нужен. Опрашиваем
        # позицию курсора таймером и пикаем сами.
        self._hover_timer = None
        if platform.system() == "Darwin":
            self._last_hover_pos = None
            self._hover_timer = QTimer(self)
            self._hover_timer.setInterval(33)  # ~30 Гц, для подсветки достаточно
            self._hover_timer.timeout.connect(self._poll_hover)
            self._hover_timer.start()

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
    #  ПИКИНГ
    # ========================

    def _pick_at(self, x, y):
        """Пикинг по координатам в виджете (логические пиксели)."""
        r = self.plotter.renderer
        ws = r.GetSize()
        w = self.plotter.interactor
        vx = int(x * ws[0] / w.width())
        vy = ws[1] - int(y * ws[1] / w.height())
        self._picker.Pick(vx, vy, 0, r)
        return self._picker.GetActor()

    def _poll_hover(self):
        """macOS-обходной путь: опросить позицию курсора и подсветить деталь.

        Зеркалит логику _on_mouse_move без зажатой кнопки, но позицию берёт из
        глобального курсора (всегда доступна), а не из недоставляемого hover-события."""
        # Во время зажатой кнопки hover не нужен: drag/rotate идёт через события.
        if QApplication.mouseButtons() != Qt.NoButton:
            return
        w = self.plotter.interactor
        if not w.isVisible():
            return
        local = w.mapFromGlobal(QCursor.pos())
        x, y = local.x(), local.y()
        if x < 0 or y < 0 or x >= w.width() or y >= w.height():
            # Курсор ушёл с 3D-сцены — один раз снимаем подсветку.
            if self._last_hover_pos is not None:
                self._last_hover_pos = None
                self.controller.on_mouse_move(None)
            return
        if (x, y) == self._last_hover_pos:
            return
        self._last_hover_pos = (x, y)
        self.controller.on_mouse_move(self._pick_at(x, y))

    # ========================
    #  МЫШЬ
    # ========================

    def _on_mouse_move(self, event):
        # Во время вращения камеры (зажата ЛКМ) не пикаем: пикинг по всей
        # модели на каждое движение делает вращение рваным, а подсветка в
        # движении не нужна.
        if self._mouse_pressed and self._mouse_button == Qt.LeftButton:
            x = event.position().x()
            dx = x - self._mouse_press_x
            self._mouse_press_x = x
            if abs(dx) > 1:
                self._mouse_has_moved = True
                self.controller.on_left_drag(dx)
            else:
                self.controller.on_left_drag(0)
            return True

        actor = self._pick_at(event.position().x(), event.position().y())
        self.controller.on_mouse_move(actor)
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
            self._mouse_pressed = False
            self._mouse_button = None
            self.controller.on_left_release()
            return True
        elif btn == Qt.RightButton:
            if self._mouse_pressed and not self._mouse_has_moved:
                self.controller.on_right_click()
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