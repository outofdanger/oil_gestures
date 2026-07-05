import numpy as np
import pyvista as pv
import vtk
from PySide6.QtGui import QImage, QPainter, QColor, QPen, QFont
from PySide6.QtCore import Qt


# ========================
# БАЗОВАЯ ДЕТАЛЬ
# ========================

class Detail:
    def __init__(self, mesh, actor, name, axis=(0, 1, 0), color=None):
        self.mesh = mesh
        self.actor = actor
        self.name = name
        self.axis = axis
        self.parent_assembly = None
        self._original_color = actor.GetProperty().GetColor()
        self.highlightable = True
        self._highlight_color = (0.0, 1.0, 0.0)

    def highlight(self):
        if self.highlightable:
            self.actor.GetProperty().SetColor(*self._highlight_color)

    def unhighlight(self):
        if self.highlightable:
            self.actor.GetProperty().SetColor(*self._original_color)

    def show(self):
        self.actor.VisibilityOn()

    def hide(self):
        self.actor.VisibilityOff()

    def get_menu_actions(self):
        return []

    def execute_action(self, action):
        pass

    def has_animation(self):
        return False

    def tick_animation(self, dt):
        pass

    @property
    def center(self):
        return self.mesh.center

    @property
    def bounds(self):
        return self.mesh.bounds

    def set_color(self, color):
        rgb = pv.Color(color).float_rgb
        self.actor.GetProperty().SetColor(*rgb)


# ========================
# ВЕНТИЛЬ
# ========================

class Valve(Detail):
    def __init__(self, mesh, actor, name, axis=(0, 1, 0), color=None):
        super().__init__(mesh, actor, name, axis)
        self._rotating = False
        self._speed = 0
        self._target = 0
        self._home = 0
        self._min = 0
        self._max = 360
        self._opened = False
        self._closed = True

    def open(self):
        if self._target == self._max:
            self.stop()
        else:
            self._target = self._max
            self._speed = 90.0
            self._rotating = True
            self._opened = True
            self._closed = False

    def close(self):
        if self._target == self._min:
            self.stop()
        else:
            self._target = self._min
            self._speed = -90.0
            self._rotating = True
            self._opened = False
            self._closed = True

    def set_position(self, percent):
        target_angle = self._max * (percent / 100)
        self._target = target_angle
        self._speed = 90.0 if target_angle > self._home else -90.0
        self._rotating = True
        self._opened = (percent == 100)
        self._closed = (percent == 0)

    def stop(self):
        self._rotating = False
        self._speed = 0

    def _rotate(self, angle):
        self.mesh.rotate_vector(self.axis, angle, point=self.center, inplace=True)
        self.mesh.Modified()

    def get_menu_actions(self):
        if self._opened:
            return [
                ("🔒 Закрыть", "close"),
                ("🔧 Частично...", "partial"),
            ]
        elif self._closed:
            return [
                ("🔓 Открыть", "open"),
                ("🔧 Частично...", "partial"),
            ]
        else:
            return [
                ("🔒 Закрыть", "close"),
                ("🔓 Открыть", "open"),
                ("🔧 Частично...", "partial"),
            ]

    def execute_action(self, action):
        if action == "open":
            self.open()
        elif action == "close":
            self.close()
        elif action.startswith("set_"):
            percent = int(action.replace("set_", ""))
            self.set_position(percent)

    def has_animation(self):
        return self._rotating

    def tick_animation(self, dt):
        if not self._rotating:
            return
        step = self._speed * dt
        if self._speed > 0 and self._home + step >= self._target:
            step = self._target - self._home
            self._rotating = False
        elif self._speed < 0 and self._home + step <= self._target:
            step = self._target - self._home
            self._rotating = False
        self._rotate(step)
        self._home += step


class Flap(Detail):
    def __init__(
            self,
            mesh,
            actor,
            name,
            color=None,
            hinge_point=None,
            air_jet=None,
            jet_origin=None,
            jet_direction=None,
    ):
        super().__init__(mesh, actor, name)
        self._animating = False
        self._phase = "idle"
        self._angle = 0.0
        self._open_angle = 24.0
        self._open_speed = 180.0
        self._close_speed = 160.0

        self._hold_time = 0.15
        self._hold_timer = 0.0

        b = self.mesh.bounds
        self._hinge_point = hinge_point or (
            b[0],
            (b[2] + b[3]) / 2,
            (b[4] + b[5]) / 2,
        )
        self._axis = (0, 1, 0)

        self._air_jet = air_jet
        self._jet_origin = np.array(jet_origin or self._hinge_point, dtype=float)
        self._jet_direction = np.array(jet_direction or (1.0, 0.0, 0.0), dtype=float)
        self._jet_started_in_hold = False

    def get_menu_actions(self):
        return [("💨 Стравить давление", "pulse_open")]

    def _start_jet(self):
        if self._air_jet is None:
            return
        self._air_jet.position = self._jet_origin.copy()
        self._air_jet.direction = self._jet_direction / np.linalg.norm(self._jet_direction)
        self._air_jet.start()

    def _stop_jet(self):
        if self._air_jet is None:
            return
        self._air_jet.stop()

    def pulse_open(self):
        if self._animating:
            return
        self._animating = True
        self._phase = "opening"
        self._hold_timer = 0.0
        self._jet_started_in_hold = False
        self._stop_jet()

    def stop(self):
        self._animating = False
        self._phase = "idle"
        self._hold_timer = 0.0
        self._jet_started_in_hold = False
        self._stop_jet()

    def execute_action(self, action):
        if action == "pulse_open":
            self.pulse_open()

    def has_animation(self):
        return self._animating

    def _rotate_to(self, target_angle):
        delta = target_angle - self._angle
        if abs(delta) < 1e-6:
            return
        self.mesh.rotate_vector(self._axis, delta, point=self._hinge_point, inplace=True)
        self.actor.GetMapper().SetInputData(self.mesh)
        self._angle = target_angle

    def tick_animation(self, dt):
        if not self._animating:
            return

        if self._phase == "opening":
            step = self._open_speed * dt
            new_angle = min(self._angle + step, self._open_angle)
            self._rotate_to(new_angle)

            if new_angle >= self._open_angle:
                self._phase = "holding"
                self._hold_timer = self._hold_time
                self._jet_started_in_hold = False

        elif self._phase == "holding":
            if not self._jet_started_in_hold:
                self._start_jet()
                self._jet_started_in_hold = True

            self._hold_timer -= dt

            if self._air_jet is not None and self._air_jet._active:
                self._air_jet.tick(dt)

            if self._hold_timer <= 0:
                self._stop_jet()
                self._phase = "closing"

        elif self._phase == "closing":
            step = self._close_speed * dt
            new_angle = max(self._angle - step, 0.0)
            self._rotate_to(new_angle)

            if new_angle <= 0.0:
                self._rotate_to(0.0)
                self._animating = False
                self._phase = "idle"


# ========================
# ЗАГЛУШКА
# ========================

class Plug(Detail):
    def __init__(self, mesh, actor, name, axis=(0, 1, 0), color=None):
        super().__init__(mesh, actor, name, axis)
        self.state = "attached"

    def remove(self):
        self.state = "removed"
        self.hide()

    def attach(self):
        self.state = "attached"
        self.show()

    def get_menu_actions(self):
        if self.state == "attached":
            return [("🔧 Снять", "remove")]
        else:
            return [("🔧 Установить", "attach")]

    def execute_action(self, action):
        if action == "remove":
            self.remove()
        elif action == "attach":
            self.attach()


# ========================
# МАНОМЕТР
# ========================

class Manometer(Detail):
    def __init__(self, mesh, actor, name, axis=(0, 1, 0), color=None):
        super().__init__(mesh, actor, name, axis)
        self._gauge_face = None
        self._gauge_arrow = None
        self._arrow_mesh = None
        self._arrow_center = None
        self._home_angle = 0
        self.state = "attached"
        self._current_mpa = 14.0

    def create_gauge(self, plotter):
        center = self.mesh.center
        pos = (center[0], center[1], center[2] + 0.054)
        size = 0.43

        plane = pv.Plane(center=pos, direction=(0, 0, 1), i_size=size, j_size=size)
        try:
            tex = pv.read_texture("assets/gauge_face.png")
            self._gauge_face = plotter.add_mesh(plane, texture=tex, opacity=1.0)
        except:
            self._gauge_face = plotter.add_mesh(plane, color="white", opacity=0.9)

        self._gauge_actor = self._gauge_face

        line = pv.Line(
            (pos[0] + 0.003, pos[1], pos[2] + 0.005),
            (pos[0] + size * 0.43, pos[1], pos[2] + 0.005)
        )
        arrow = line.tube(radius=0.0036)
        tip = pv.Sphere(radius=0.015, center=(pos[0], pos[1], pos[2]))
        arrow = arrow.merge(tip)
        self._gauge_arrow = plotter.add_mesh(arrow, color="red")
        self._arrow_mesh = arrow
        self._arrow_center = (pos[0], pos[1], pos[2] + 0.005)
        self.set_pressure_mpa(0.0)

    def set_pressure_mpa(self, mpa):
        mpa = max(0, min(16, mpa))
        if abs(mpa - self._current_mpa) < 0.01:  # не изменилось
            return
        self._current_mpa = mpa
        percent = mpa / 16 * 100
        angle = 210 - 240 * (percent / 100)
        delta = angle - self._home_angle
        if self._arrow_mesh:
            self._arrow_mesh.rotate_vector((0, 0, 1), delta, point=self._arrow_center, inplace=True)
            self._gauge_arrow.GetMapper().SetInputData(self._arrow_mesh)
        self._home_angle = angle

    def highlight(self):
        super().highlight()
        if self._gauge_face:
            self._gauge_face.GetProperty().SetColor(1.0, 1.0, 0.0)

    def unhighlight(self):
        super().unhighlight()
        if self._gauge_face:
            self._gauge_face.GetProperty().SetColor(1.0, 1.0, 1.0)

    def remove(self):
        self.state = "removed"
        self.hide()
        self._current_mpa = 0.0
        self.set_pressure_mpa(0.0)
        if self._gauge_face:
            self._gauge_face.VisibilityOff()
        if self._gauge_arrow:
            self._gauge_arrow.VisibilityOff()

    def attach(self):
        self.state = "attached"
        self.show()
        if self._gauge_face:
            self._gauge_face.VisibilityOn()
        if self._gauge_arrow:
            self._gauge_arrow.VisibilityOn()

    def get_menu_actions(self):
        if self.state == "attached":
            return [
                ("🔧 Снять", "remove"),
            ]
        else:
            return [("🔧 Установить", "attach")]

    def execute_action(self, action):
        if action == "remove":
            self.remove()
        elif action == "attach":
            self.attach()

    @property
    def pressure_mpa(self):
        return self._current_mpa


# ========================
# КОРПУС
# ========================

class Body(Detail):
    def __init__(self, mesh, actor, name, axis=(0, 1, 0), color=None):
        super().__init__(mesh, actor, name, axis)
        self.highlightable = False


class TouchScreen:
    """
    Миксин для LevelGaugeScreen/ControllerScreen: делает текстурный экран
    полноценно кликабельным. Хранит геометрию плоскости экрана и
    пиксельные прямоугольники каждой отрисованной строки (см.
    render_lines), плюс карту "индекс строки -> что нажать".

    Никакой логики приборов тут нет — только перевод 3D-точки клика в
    пиксель текстуры и обратный поиск того, какая физическая кнопка (или
    последовательность кнопок) должна быть виртуально "нажата".
    """

    def _init_touch_screen(self):
        self._plane_center = np.zeros(3)
        self._plane_half_w = 0.5
        self._plane_half_h = 0.5
        self._line_rects = []   # [(x0, y0, x1, y1), ...] по индексу строки
        self._regions = {}      # {line_index: "button_name" | ("button_name", ...)}

    def _set_plane_geometry(self, center, width, height):
        self._plane_center = np.array(center, dtype=float)
        self._plane_half_w = max(width, 1e-6) / 2.0
        self._plane_half_h = max(height, 1e-6) / 2.0

    def set_regions(self, regions):
        """
        regions: {line_index: "имя_детали"} для прямого соответствия
        1 строка = 1 кнопка, или {line_index: ("имя_1", "имя_2", ...)} для
        последовательности виртуальных нажатий (например, чтобы долистать
        до нужного пункта меню и подтвердить его одним тапом).
        """
        self._regions = dict(regions) if regions else {}

    def world_to_pixel(self, world_point):
        local = np.array(world_point, dtype=float) - self._plane_center
        u = (local[0] + self._plane_half_w) / (2 * self._plane_half_w)
        v = 1.0 - (local[1] + self._plane_half_h) / (2 * self._plane_half_h)
        return u * self._tex_w, v * self._tex_h

    def hit_test(self, world_point):
        if not self._regions:
            return None
        px, py = self.world_to_pixel(world_point)
        for index, target in self._regions.items():
            if index >= len(self._line_rects):
                continue
            x0, y0, x1, y1 = self._line_rects[index]
            if x0 <= px <= x1 and y0 <= py <= y1:
                return target
        return None


class LevelGaugeScreen(TouchScreen, Detail):
    def __init__(self, mesh, actor, name, plotter, color=None):
        super().__init__(mesh, actor, name)
        self._init_touch_screen()
        self.plotter = plotter
        self.highlightable = False
        self._screen_plane_mesh = None
        self._screen_plane_actor = None
        self._texture = None
        self._tex_w = 512
        self._tex_h = 512
        self._font_scale = 0.95
        self._build_plane()
        self.render_lines(["УРОВНЕМЕР"])

    def _build_plane(self):
        b = self.mesh.bounds
        center = (
            (b[0] + b[1]) / 2,
            (b[2] + b[3]) / 2,
            b[5] + 0.002,
        )
        width = max(b[1] - b[0], 0.01)
        height = max(b[3] - b[2], 0.01)

        # Match texture aspect ratio to the physical screen dimensions
        # Use 512 on the longer axis so both axes are high-res
        if width >= height:
            self._tex_w = 512
            self._tex_h = max(64, int(512 * height / width))
        else:
            self._tex_h = 512
            self._tex_w = max(64, int(512 * width / height))

        self._screen_plane_mesh = pv.Plane(
            center=center,
            direction=(0, 0, 1),
            i_size=width * 0.94,
            j_size=height * 0.94,
        )
        self._set_plane_geometry(center, width * 0.94, height * 0.94)

        self._screen_plane_actor = self.plotter.add_mesh(
            self._screen_plane_mesh,
            color="black",
            lighting=False,
            opacity=1.0,
        )

    def render_lines(self, lines):
        w, h = self._tex_w, self._tex_h

        image = QImage(w, h, QImage.Format_RGBA8888)
        image.fill(QColor(18, 28, 18))

        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Background with a subtle border
        painter.fillRect(0, 0, w, h, QColor(20, 45, 20))
        painter.setPen(QPen(QColor(60, 100, 60), 3))
        painter.drawRect(4, 4, w - 8, h - 8)

        # Scale font and spacing relative to texture height
        max_lines = 6
        padding_top = int(h * 0.10)
        padding_left = int(w * 0.05)
        line_height = int((h - padding_top * 2) / max_lines)
        font_size = max(6, int(line_height * self._font_scale))

        font = QFont("Consolas", font_size)
        font.setBold(False)
        painter.setFont(font)

        is_header = True
        # Сбрасываем старые кликабельные зоны — если после этого рендера
        # никто не вызовет set_regions(), экран просто не будет
        # интерактивным (безопасное поведение по умолчанию), а не будет
        # ссылаться на строки предыдущего состояния экрана
        self._regions = {}
        self._line_rects = []

        y = padding_top + line_height
        for line in lines[:max_lines]:
            # Прямоугольник строки для тач-хиттеста фиксируем ДО проверки
            # на пустую строку, чтобы индексы совпадали с исходным lines
            self._line_rects.append((padding_left, y - line_height, w - padding_left, y))

            if line.strip() == "":
                y += line_height
                continue

            if is_header:
                # Header line: brighter, bold
                header_font = QFont("Consolas", font_size)
                header_font.setBold(True)
                painter.setFont(header_font)
                painter.setPen(QPen(QColor(180, 255, 180)))
                is_header = False
            elif line.startswith(">"):
                # Selected menu item: highlighted
                painter.setFont(QFont("Consolas", font_size))
                painter.setPen(QPen(QColor(255, 255, 100)))
            else:
                painter.setFont(QFont("Consolas", font_size))
                painter.setPen(QPen(QColor(120, 230, 120)))

            painter.drawText(padding_left, y, line)
            y += line_height

        painter.end()

        ptr = image.bits()
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4)).copy()

        self._texture = pv.Texture(arr)

        if self._screen_plane_actor is not None:
            self.plotter.remove_actor(self._screen_plane_actor, render=False)

        self._screen_plane_actor = self.plotter.add_mesh(
            self._screen_plane_mesh,
            texture=self._texture,
            lighting=False,
            opacity=1.0,
        )

        # If the assembly is currently removed/hidden, keep this actor hidden too
        if self.parent_assembly is not None and self.parent_assembly.state == "removed":
            self._screen_plane_actor.VisibilityOff()

    def show(self):
        super().show()
        if self._screen_plane_actor:
            self._screen_plane_actor.VisibilityOn()

    def hide(self):
        super().hide()
        if self._screen_plane_actor:
            self._screen_plane_actor.VisibilityOff()


class LevelGaugeCover(Detail):
    """
    Крышка уровнемера — открывается/закрывается по ЛКМ.
    При закрытии скрывает экран, при открытии показывает.
    """

    def __init__(self, mesh, actor, name, screen_detail=None, color=None):
        super().__init__(mesh, actor, name)
        self.highlightable = True
        self._screen_detail = screen_detail

        self._angle = 0.0
        self._target_angle = 0.0
        self._speed = 100.0
        self._animating = False
        self._closed = False

        b = self.mesh.bounds
        # Петля — верхний край крышки (y_max), крышка складывается вниз
        # Точка вращения берётся по переднему краю (z_max) верхнего ребра
        self._hinge_point = (
            (b[0] + b[1]) / 2,
            b[2],  # y_max — верхний край, он неподвижен
            (b[4]),
        )
        # Ось Z: крышка поворачивается вперёд/назад относительно верхнего края
        self._axis = (1, 0, 0)

    def set_screen(self, screen_detail):
        self._screen_detail = screen_detail

    def open(self):
        if self._animating or not self._closed:
            return
        self._target_angle = 0.0
        self._closed = False
        self._animating = True
        if self._screen_detail is not None:
            self._screen_detail.show()

    def close(self):
        if self._animating or self._closed:
            return
        self._target_angle = 133.9
        self._closed = True
        self._animating = True

    def get_menu_actions(self):
        if self._closed:
            return [("🔓 Открыть крышку", "open")]
        return [("🔒 Закрыть крышку", "close")]

    def execute_action(self, action):
        if action == "open":
            self.open()
        elif action == "close":
            self.close()

    def has_animation(self):
        return self._animating

    def tick_animation(self, dt):
        if not self._animating:
            return

        delta = self._target_angle - self._angle
        if abs(delta) < 0.5:
            self._apply_rotation(delta)
            self._animating = False
            if self._closed and self._screen_detail is not None:
                self._screen_detail.hide()
            return

        step = self._speed * dt
        if delta < 0:
            step = -step
        if abs(step) > abs(delta):
            step = delta

        self._apply_rotation(step)

    def _apply_rotation(self, step):
        self.mesh.rotate_vector(self._axis, step, point=self._hinge_point, inplace=True)
        self.actor.GetMapper().SetInputData(self.mesh)
        self._angle += step


class TexturedButton(Detail):
    def __init__(
            self,
            mesh,
            actor,
            name,
            plotter,
            label="MODE",
            bg_color=(190, 190, 190, 255),
            text_color=(20, 20, 20, 255),
            border_color=(110, 110, 110, 255),
            font_family="Arial",
            font_scale=0.2,
    ):
        super().__init__(mesh, actor, name)
        self.plotter = plotter
        self.label = label
        self.bg_color = bg_color
        self.text_color = text_color
        self.border_color = border_color
        self.font_family = font_family
        self.font_scale = font_scale
        self.texture_actor = None
        self.texture = None
        self._plane = None
        self.create_texture()

    def create_texture(self):
        b = self.mesh.bounds
        center = self.mesh.center
        width = max(b[1] - b[0], 0.01)
        height = max(b[3] - b[2], 0.01)

        self._plane = pv.Plane(
            center=(center[0], center[1], b[5] + 0.003),
            direction=(0, 0, 1),
            i_size=width * 0.90,
            j_size=height * 0.90,
        )

        if width >= height:
            tex_w = 512
            tex_h = max(128, int(512 * height / width))
        else:
            tex_h = 512
            tex_w = max(128, int(512 * width / height))

        image = QImage(tex_w, tex_h, QImage.Format_RGBA8888)
        image.fill(QColor(*self.bg_color))

        painter = QPainter(image)

        # Для более резкого текста:
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setRenderHint(QPainter.TextAntialiasing, False)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)

        painter.fillRect(0, 0, tex_w, tex_h, QColor(*self.bg_color))
        painter.setPen(QPen(QColor(*self.border_color), 2))
        painter.drawRect(1, 1, tex_w - 2, tex_h - 2)

        font_px = max(8, int(min(tex_w, tex_h) * self.font_scale))
        font = QFont(self.font_family)
        font.setPixelSize(font_px)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(*self.text_color))

        text_rect = image.rect().adjusted(8, 4, -8, -4)
        painter.drawText(text_rect, Qt.AlignCenter | Qt.AlignVCenter, self.label)

        painter.end()

        ptr = image.bits()
        arr = np.frombuffer(ptr, np.uint8).reshape((tex_h, tex_w, 4)).copy()
        self.texture = pv.Texture(arr)

        # Важно: отключить интерполяцию текстуры
        self.texture.interpolate = False

        if self.texture_actor is not None:
            self.plotter.remove_actor(self.texture_actor, render=False)

        self.texture_actor = self.plotter.add_mesh(
            self._plane,
            texture=self.texture,
            lighting=False,
            opacity=1.0,
            show_edges=False,
        )

        self.texture_actor.PickableOff()

    def show(self):
        super().show()
        if self.texture_actor is not None:
            self.texture_actor.VisibilityOn()

    def hide(self):
        super().hide()
        if self.texture_actor is not None:
            self.texture_actor.VisibilityOff()


# ========================
# СБОРКА УРОВНЕМЕРА
# ========================

class LevelGaugeAssembly:
    def __init__(self, name, parts, emitters=None):
        self.name = name
        self.parts = parts
        self.emitters = emitters or []
        self.state = "removed"
        self.highlightable = False
        self.actor = None
        self.mesh = None

        for part in self.parts:
            part.parent_assembly = self
            part.hide()

        for emitter in self.emitters:
            emitter.hide()

    def show(self):
        for part in self.parts:
            part.show()

    def hide(self):
        for part in self.parts:
            part.hide()

        for emitter in self.emitters:
            emitter.stop()
            emitter.hide()

    def attach(self):
        self.state = "attached"
        self.show()

    def remove(self):
        self.state = "removed"
        self.hide()

    @property
    def bounds(self):
        xs = []
        ys = []
        zs = []
        for part in self.parts:
            b = part.bounds
            xs.extend([b[0], b[1]])
            ys.extend([b[2], b[3]])
            zs.extend([b[4], b[5]])
        return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))

    def get_menu_actions(self):
        if self.state == "attached":
            return [("🔧 Снять", "remove")]
        return [("🔧 Установить", "attach")]

    def execute_action(self, action):
        if action == "attach":
            self.attach()
        elif action == "remove":
            self.remove()

    def has_animation(self):
        return False

    def tick_animation(self, dt):
        pass

    def highlight(self):
        pass

    def unhighlight(self):
        pass


# ========================
# ЧАСТИЦЫ
# ========================

class ParticleSystem:
    OIL = "oil"
    GAS = "gas"
    AIR_BLAST = "air_blast"

    def __init__(self, plotter, position, direction=(0, 1, 0), particle_type=OIL, count=400):
        self.plotter = plotter
        self.position = np.array(position, dtype=float)
        self.direction = np.array(direction, dtype=float)
        self.direction = self.direction / np.linalg.norm(self.direction)
        self._type = particle_type
        self._count = count
        self._active = False
        self._base_point_size = 8.0
        self._min_point_size = 6.0
        self._max_point_size = 18.0
        self._reference_distance = None
        self._base_count = count
        self._active_count = count
        self._max_count = count
        self._frame_counter = 0

        if particle_type == self.OIL:
            # Нефть: не чистый чёрный, а тёмный коричнево-оливковый —
            # так блик на поверхности выглядит контрастнее и реалистичнее
            self._color = (0.03, 0.02, 0.015)
            self._opacity = 0.92
            self._gravity = np.array([0, -3.0, 0])
            self._lifetime_min = 1.5
            self._lifetime_max = 2.5
            self._speed_min = 1.2
            self._speed_max = 2.5
            # база для размера блика (см. ниже) — крупнее, чем раньше,
            # чтобы блик оставался заметным на увеличенных каплях
            self._base_point_size = 16.0
            self._min_point_size = 12.0
            self._max_point_size = 22.0
            self._glare_base_opacity = 0.0
            self._glare_flicker_timer = 0.0
            # Реальный физический размер капли в единицах сцены (не пиксели!).
            # Увеличен в несколько раз, чтобы капли перекрывали друг друга
            # и читались как единая жидкость, а не отдельные точки.
            # Подберите под масштаб вашей модели.
            self._droplet_world_size = 0.06
            # Пересборка мешей капель (vtkGlyph3D) — самая дорогая операция
            # в tick(). 1 = каждый кадр (максимально плавно, но дороже всего).
            # 2-3 = обновлять раз в 2-3 кадра — заметно снижает нагрузку
            # на CPU/GPU почти без потери плавности на глаз.
            self._glyph_update_every = 2
            self._glyph_frame_counter = 0
        elif particle_type == self.AIR_BLAST:
            self._color = "white"
            self._opacity = 0.25
            self._speed_min = 20.0
            self._speed_max = 30.0
            self._lifetime_min = 0.60
            self._lifetime_max = 1.00
            self._gravity = np.array([0.0, -0.2, 0.0])
            self._base_point_size = 8.0
            self._min_point_size = 6.0
            self._max_point_size = 18.0
        else:  # GAS
            self._color = (0.85, 0.85, 0.85)  # светло-серый
            self._opacity = 0.15
            self._gravity = np.array([0, -0.3, 0])
            self._lifetime_min = 0.3
            self._lifetime_max = 1.0
            self._speed_min = 5.0
            self._speed_max = 9.0
            self._base_point_size = 4.0
            self._min_point_size = 2.0
            self._max_point_size = 8.0

        self._positions = np.tile(self.position, (count, 1))
        self._velocities = np.zeros((count, 3))
        self._lifetimes = np.random.uniform(0, 2.0, count)
        self._mesh = pv.PolyData(self._positions)

        if particle_type == self.OIL:
            # Небольшой случайный разброс цвета/яркости по каждой капле —
            # однородный цвет по всей массе точек выдаёт "частицы",
            # а лёгкая неоднородность делает жидкость правдоподобнее
            base = np.array(self._color)
            variation = np.random.uniform(0.7, 1.3, (count, 1))
            per_point_colors = np.clip(base * variation, 0.0, 1.0)
            self._mesh["oil_colors"] = per_point_colors

            # Форма капли (не сфера): округлый "нос" по направлению падения
            # и заострённый "хвост" сзади — ориентируется по вектору скорости.
            # Уже/тоньше, чем раньше (меньше радиус относительно длины) —
            # такая пропорция сильнее читается как жидкая капля, а не шарик
            self._droplet_template = self._build_droplet_template(length=1.3, max_radius=0.4)
            self._droplet_scale_base = np.random.uniform(1.0, 1.7, count)
            self._mesh["droplet_scale"] = self._droplet_scale_base.copy()
            # изначально капли "смотрят" вниз, пока не появится скорость
            self._mesh["velocity_dir"] = np.tile([0.0, -1.0, 0.0], (count, 1))
            self._mesh.set_active_scalars("droplet_scale")
            self._mesh.set_active_vectors("velocity_dir")

            self._glyph_filter = vtk.vtkGlyph3D()
            self._glyph_filter.SetSourceData(self._droplet_template)
            self._glyph_filter.SetInputData(self._mesh)
            self._glyph_filter.OrientOn()
            self._glyph_filter.SetVectorModeToUseVector()
            self._glyph_filter.SetScaleModeToScaleByScalar()
            self._glyph_filter.SetScaleFactor(self._droplet_world_size)
            self._glyph_filter.Update()

            self._actor = plotter.add_mesh(
                pv.wrap(self._glyph_filter.GetOutput()),
                scalars="oil_colors",
                rgb=True,
                opacity=self._opacity,
                lighting=True,
                smooth_shading=True,
            )
        else:
            self._actor = plotter.add_mesh(
                self._mesh,
                color=self._color,
                point_size=self._base_point_size,
                render_points_as_spheres=True,
                opacity=self._opacity,
                lighting=True,
                smooth_shading=True,
            )
        self._actor.VisibilityOff()

        # Настройка материала для блеска (особенно для нефти)
        prop = self._actor.GetProperty()
        if self._type == self.OIL:
            # PBR даёт куда более честный, "мокрый" блик, чем классический
            # Phong-specular, при этом резкость блика задаёт Roughness,
            # а не только SpecularPower. Для полного эффекта в сцене должна
            # быть задана environment texture, см. _apply_oil_environment().
            # Roughness понижен — чем он ближе к 0, тем более зеркальным
            # и резким выглядит блик (0.3–0.4 дало бы более матовую нефть)
            prop.SetInterpolationToPBR()
            prop.SetMetallic(0.0)
            prop.SetRoughness(0.3)
            prop.SetSpecular(1.0)
            prop.SetSpecularColor(1.0, 1.0, 1.0)
            prop.SetDiffuse(0.5)
        else:
            prop.SetSpecular(0.1)
            prop.SetSpecularPower(5)
            prop.SetDiffuse(0.8)

        # Второй, "бликующий" слой поверх тех же точек: маленькие, очень
        # яркие спрайты без освещения — имитируют резкий солнечный/студийный
        # блик на поверхности капли, который PBR один не всегда красиво
        # даёт на мелких сферах. По умолчанию невидим (opacity=0),
        # включается мерцанием в tick().
        self._glare_actor = None
        if particle_type == self.OIL:
            self._glare_actor = plotter.add_mesh(
                self._mesh,
                color=(1.0, 1.0, 1.0),
                point_size=self._base_point_size * 0.35,
                render_points_as_spheres=True,
                opacity=0.0,
                lighting=False,
            )
            self._glare_actor.VisibilityOff()

    def _build_droplet_template(self, length=1.0, max_radius=0.45, resolution=6):
        """
        Строит меш капли как поверхность вращения профиля вокруг оси X:
        узкий заострённый "хвост" при x=0 и округлый "нос" при x=length.
        vtkGlyph3D по умолчанию ориентирует источник так, что его локальная
        ось X совпадает с вектором (в нашем случае — с вектором скорости),
        поэтому "нос" капли будет направлен вперёд по движению, а "хвост"
        останется вытянутым позади — как у настоящей падающей капли.

        ВАЖНО ДЛЯ ПРОИЗВОДИТЕЛЬНОСТИ: этот меш перестраивается через
        vtkGlyph3D на КАЖДОЙ капле КАЖДЫЙ кадр (см. tick()), поэтому число
        точек профиля и resolution напрямую умножаются на количество
        частиц — при 500 каплях даже небольшое увеличение здесь ощутимо
        нагружает CPU/GPU. Текущие значения — компромисс между "видно, что
        это капля" и низкой нагрузкой; поднимайте resolution только если
        точно есть запас по производительности.
        """
        neck = length * 0.55
        n_tail, n_nose = 5, 5

        x_tail = np.linspace(0.0, neck, n_tail)
        r_tail = max_radius * (x_tail / neck)

        x_nose = np.linspace(neck, length, n_nose)[1:]
        t = (x_nose - neck) / (length - neck)
        r_nose = max_radius * np.sqrt(np.clip(1.0 - t ** 2, 0.0, 1.0))

        x = np.concatenate([x_tail, x_nose])
        r = np.concatenate([r_tail, r_nose])
        r[0] = 0.0
        r[-1] = 0.0

        profile_points = np.column_stack([x, r, np.zeros_like(r)])
        n = len(profile_points)
        profile = pv.PolyData(profile_points)
        profile.lines = np.hstack([[n], np.arange(n)])

        droplet = profile.extrude_rotate(resolution=resolution, capping=True, rotation_axis=(1, 0, 0))
        droplet.translate([-length / 2.0, 0.0, 0.0], inplace=True)
        droplet.compute_normals(auto_orient_normals=True, inplace=True)
        return droplet

    def apply_oil_environment(self, cubemap=None):
        """
        PBR-материал нефти (Metallic/Roughness) даёт по-настоящему
        реалистичный блик только тогда, когда у рендерера есть environment
        texture — иначе блики будут почти незаметны, т.к. PBR освещает
        поверхность отражением окружения, а не только точечными источниками.

        Вызвать один раз после создания плоттера, например:
            oil_system.apply_oil_environment()
        или передать свой кубмап:
            cubemap = pv.cubemap(path="assets/skybox")  # 6 картинок
            oil_system.apply_oil_environment(cubemap)
        """
        if self._type != self.OIL:
            return
        if cubemap is None:
            cubemap = pv.examples.download_sky_box_cube_map()
        self.plotter.set_environment_texture(cubemap)

    def show(self):
        self._actor.VisibilityOn()
        if self._glare_actor is not None:
            self._glare_actor.VisibilityOn()

    def hide(self):
        self._actor.VisibilityOff()
        if self._glare_actor is not None:
            self._glare_actor.VisibilityOff()

    def start(self):
        self._active = True
        self._lifetimes = np.random.uniform(0, self._lifetime_max, self._count)
        self._actor.VisibilityOn()
        if self._glare_actor is not None:
            self._glare_actor.VisibilityOn()

    def stop(self):
        self._active = False
        self._positions[:] = self.position
        self._velocities[:] = 0
        self._lifetimes[:] = 0
        self._mesh.points = self._positions
        if self._type == self.OIL:
            self._mesh["droplet_scale"] = np.zeros(self._count)
            self._glyph_filter.Update()
            self._actor.GetMapper().SetInputData(self._glyph_filter.GetOutput())
        else:
            self._actor.GetMapper().SetInputData(self._mesh)
        self._actor.VisibilityOff()
        if self._glare_actor is not None:
            self._glare_actor.GetMapper().SetInputData(self._mesh)
            self._glare_actor.GetProperty().SetOpacity(0.0)
            self._glare_actor.VisibilityOff()

    def set_intensity(self, percent):
        percent = max(5, min(100, percent))
        self._active_count = int(self._base_count * percent / 100)

    def _update_point_size_by_camera(self):
        camera = getattr(self.plotter, "camera", None)
        if camera is None:
            return
        cam_pos = np.array(camera.position, dtype=float)
        center = self._positions.mean(axis=0)
        distance = np.linalg.norm(cam_pos - center)
        if distance < 1e-6:
            return
        if self._reference_distance is None:
            self._reference_distance = distance
        scale = self._reference_distance / distance
        new_size = self._base_point_size * scale
        new_size = max(self._min_point_size, min(self._max_point_size, new_size))

        if self._type != self.OIL:
            # Капли-нефть теперь реальная геометрия в мировых единицах, а не
            # экранные point-спрайты — её масштабировать по камере не нужно,
            # перспектива и так делает объекты меньше на расстоянии.
            self._actor.GetProperty().SetPointSize(new_size)

        if self._glare_actor is not None:
            self._glare_actor.GetProperty().SetPointSize(new_size * 0.35)

    def tick(self, dt=0.016):
        if not self._active:
            return

        n = self._count
        idx = np.arange(n)
        active_mask = idx < self._active_count
        inactive_mask = ~active_mask

        # частицы вне текущей интенсивности — держим "свёрнутыми" в источнике
        if inactive_mask.any():
            self._lifetimes[inactive_mask] = 0
            self._positions[inactive_mask] = self.position
            self._velocities[inactive_mask] = 0

        respawn_mask = active_mask & (self._lifetimes <= 0)
        n_respawn = int(respawn_mask.sum())
        if n_respawn:
            if self._type == self.AIR_BLAST:
                angle1 = np.random.uniform(0, np.pi / 120, n_respawn)
            else:
                angle1 = np.random.uniform(0, np.pi / 12, n_respawn)
            angle2 = np.random.uniform(0, 2 * np.pi, n_respawn)
            dirs = self._rotate_cone_batch(angle1, angle2)
            speeds = np.random.uniform(self._speed_min, self._speed_max, n_respawn)

            new_pos = np.tile(self.position, (n_respawn, 1))
            new_pos[:, 1] = np.maximum(new_pos[:, 1], 0.0)
            self._positions[respawn_mask] = new_pos
            self._velocities[respawn_mask] = dirs * speeds[:, None]
            self._lifetimes[respawn_mask] = np.random.uniform(
                self._lifetime_min, self._lifetime_max, n_respawn
            )

        alive_mask = active_mask & ~respawn_mask
        if alive_mask.any():
            moving_mask = alive_mask
            if self._type != self.AIR_BLAST:
                speeds_now = np.linalg.norm(self._velocities, axis=1)
                stuck_mask = alive_mask & (speeds_now < 1e-9)
                if stuck_mask.any():
                    self._lifetimes[stuck_mask] -= dt
                moving_mask = alive_mask & ~stuck_mask

            if moving_mask.any():
                self._velocities[moving_mask] += self._gravity * dt
                new_pos = self._positions[moving_mask] + self._velocities[moving_mask] * dt
                floor_hit = new_pos[:, 1] < 0.0
                new_pos[floor_hit, 1] = 0.0
                vel_slice = self._velocities[moving_mask]
                vel_slice[floor_hit, 1] = 0.0
                self._velocities[moving_mask] = vel_slice
                self._positions[moving_mask] = new_pos
                self._lifetimes[moving_mask] -= dt

        self._mesh.points = self._positions

        if self._type == self.OIL:
            # Ориентируем каждую каплю по направлению её текущей скорости
            # (vtkGlyph3D совмещает локальную ось X шаблона с этим вектором),
            # у только что заспавненных/зависших капель скорость нулевая —
            # подставляем направление "вниз", чтобы не было вырожденного вектора
            speeds = np.linalg.norm(self._velocities, axis=1, keepdims=True)
            dirs = np.divide(
                self._velocities, speeds,
                out=np.tile([0.0, -1.0, 0.0], (self._count, 1)),
                where=speeds > 1e-6,
            )
            self._mesh["velocity_dir"] = dirs

            # Капли за пределами текущей интенсивности (active_count) прячем
            # через нулевой масштаб, а не оставляем "слипшимися" в источнике
            scales = self._droplet_scale_base.copy()
            scales[self._active_count:] = 0.0
            self._mesh["droplet_scale"] = scales

            # Самая тяжёлая часть — пересборка треугольной геометрии капель.
            # Делаем это не каждый кадр, а раз в _glyph_update_every кадров
            self._glyph_frame_counter += 1
            if self._glyph_frame_counter >= self._glyph_update_every:
                self._glyph_frame_counter = 0
                self._glyph_filter.Update()
                self._actor.GetMapper().SetInputData(self._glyph_filter.GetOutput())
        else:
            self._actor.GetMapper().SetInputData(self._mesh)

        if self._glare_actor is not None:
            self._glare_actor.GetMapper().SetInputData(self._mesh)
            # Мерцание блика: случайно "вспыхивает" и гаснет, как солнечный
            # зайчик на поверхности движущихся капель, а не статичная точка
            self._glare_flicker_timer -= dt
            if self._glare_flicker_timer <= 0.0:
                self._glare_base_opacity = np.random.uniform(0.0, 0.7)
                self._glare_flicker_timer = np.random.uniform(0.05, 0.15)
            self._glare_actor.GetProperty().SetOpacity(self._glare_base_opacity)

        self._frame_counter += 1
        if self._frame_counter == 10:
            self._frame_counter = 0
            self._update_point_size_by_camera()

    def _rotate_cone(self, v, angle1, angle2):
        if abs(v[0]) < 0.001 and abs(v[2]) < 0.001:
            perp = np.array([1, 0, 0])
        else:
            perp = np.array([-v[2], 0, v[0]])
        perp = perp / np.linalg.norm(perp)
        v = self._rot(v, perp, angle1)
        v = self._rot(v, np.array(self.direction), angle2)
        return v

    def _rotate_cone_batch(self, angle1, angle2):
        """
        Векторизованная версия _rotate_cone: считает направления сразу для
        массива частиц (angle1/angle2 — массивы одинаковой длины), без
        Python-цикла. self.direction — общая константа для всех частиц.
        """
        v = np.asarray(self.direction, dtype=float)
        if abs(v[0]) < 0.001 and abs(v[2]) < 0.001:
            perp = np.array([1.0, 0.0, 0.0])
        else:
            perp = np.array([-v[2], 0.0, v[0]])
        perp = perp / np.linalg.norm(perp)

        cos1 = np.cos(angle1)[:, None]
        sin1 = np.sin(angle1)[:, None]
        v1 = v * cos1 + np.cross(perp, v) * sin1 + perp * np.dot(perp, v) * (1 - cos1)

        axis2 = v / np.linalg.norm(v)
        cos2 = np.cos(angle2)[:, None]
        sin2 = np.sin(angle2)[:, None]
        dot2 = (v1 * axis2).sum(axis=1, keepdims=True)
        v2 = v1 * cos2 + np.cross(axis2, v1) * sin2 + axis2 * dot2 * (1 - cos2)
        return v2

    def _rot(self, v, axis, angle):
        axis = axis / np.linalg.norm(axis)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return v * cos_a + np.cross(axis, v) * sin_a + axis * np.dot(axis, v) * (1 - cos_a)


class ControllerDoor(Detail):
    def __init__(self, mesh, actor, name, color=None, hinge_point=None):
        super().__init__(mesh, actor, name, color)
        self._angle = 0.0
        self._target_angle = 0.0
        self._speed = 120.0
        self._animating = False
        self._opened = True

        b = self.mesh.bounds
        self._hinge_point = hinge_point or (
            b[0],
            (b[2] + b[3]) / 2,
            (b[4] + b[5]) / 2,
        )
        self._axis = (0, 1, 0)

    def open(self):
        self._target_angle = 0.0
        self._opened = True
        self._animating = True

    def close(self):
        self._target_angle = 90.0
        self._opened = False
        self._animating = True

    def get_menu_actions(self):
        if self._opened:
            return [("Закрыть дверцу", "close")]
        return [("Открыть дверцу", "open")]

    def execute_action(self, action):
        if action == "open":
            self.open()
        elif action == "close":
            self.close()

    def has_animation(self):
        return self._animating

    def tick_animation(self, dt):
        if not self._animating:
            return

        delta = self._target_angle - self._angle

        if abs(delta) < 0.5:
            self._rotate_to(self._target_angle)
            self._animating = False
            return

        step = self._speed * dt
        if delta < 0:
            step = -step

        if abs(step) > abs(delta):
            step = delta

        self.mesh.rotate_vector(self._axis, step, point=self._hinge_point, inplace=True)
        self.actor.GetMapper().SetInputData(self.mesh)
        self._angle += step

    def _rotate_to(self, target_angle):
        delta = target_angle - self._angle
        if abs(delta) < 1e-6:
            return
        self.mesh.rotate_vector(self._axis, delta, point=self._hinge_point, inplace=True)
        self.actor.GetMapper().SetInputData(self.mesh)
        self._angle = target_angle


class ControllerLever(Detail):
    def __init__(self, mesh, actor, name, color=None, pivot_point=None):
        super().__init__(mesh, actor, name, color)
        self._angle = 0.0
        self._target_angle = 0.0
        self._speed = 160.0
        self._animating = False
        self._on = False

        b = self.mesh.bounds
        self._pivot_point = pivot_point or (
            (b[0] + b[1]) / 2,
            b[2],
            b[4],
        )

        # Ось проходит вдоль ширины рычага; pivot стоит на нижнем крае крепления.
        self._axis = (1, 0, 0)

    def toggle(self):
        self._on = not self._on
        self._target_angle = 80.0 if self._on else 0.0
        self._animating = True

    def force_off(self):
        self._on = False
        self._target_angle = 0.0
        self._rotate_to(0.0)
        self._animating = False

    def execute_action(self, action):
        if action == "toggle":
            self.toggle()
        elif action == "force_off":
            self.force_off()

    def has_animation(self):
        return self._animating

    def tick_animation(self, dt):
        if not self._animating:
            return

        delta = self._target_angle - self._angle
        if abs(delta) < 0.5:
            self._rotate_to(self._target_angle)
            self._animating = False
            return

        step = self._speed * dt
        if delta < 0:
            step = -step
        if abs(step) > abs(delta):
            step = delta

        self.mesh.rotate_vector(self._axis, step, point=self._pivot_point, inplace=True)
        self.actor.GetMapper().SetInputData(self.mesh)
        self._angle += step

    def _rotate_to(self, target_angle):
        delta = target_angle - self._angle
        if abs(delta) < 1e-6:
            return
        self.mesh.rotate_vector(self._axis, delta, point=self._pivot_point, inplace=True)
        self.actor.GetMapper().SetInputData(self.mesh)
        self._angle = target_angle


class ControllerScreen(LevelGaugeScreen):
    def __init__(self, mesh, actor, name, plotter, color=None):
        super().__init__(mesh, actor, name, plotter, color)
        self._font_scale = 0.5
        self.render_lines(["КОНТРОЛЛЕР", "", "ПИТАНИЕ: ВЫКЛ"])

    def _build_plane(self):
        b = self.mesh.bounds
        center = (
            (b[0] + b[1]) / 2,
            (b[2] + b[3]) / 2,
            b[5] + 0.006,
        )

        width = max(b[1] - b[0], 0.01)
        height = max(b[3] - b[2], 0.01)

        # Вот эти множители увеличивают видимый экран
        width_scale = 2
        height_scale = 1.15

        if width >= height:
            self._tex_w = 512
            self._tex_h = max(64, int(512 * height / width))
        else:
            self._tex_h = 512
            self._tex_w = max(64, int(512 * width / height))

        self._screen_plane_mesh = pv.Plane(
            center=center,
            direction=(0, 0, 1),
            i_size=width * width_scale,
            j_size=height * height_scale,
        )
        self._set_plane_geometry(center, width * width_scale, height * height_scale)

        self._screen_plane_actor = self.plotter.add_mesh(
            self._screen_plane_mesh,
            color="black",
            lighting=False,
            opacity=1.0,
        )