import numpy as np
import pyvista as pv
from PySide6.QtGui import QImage, QPainter, QColor, QPen, QFont
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

    def open(self):
        if self._target == self._max:
            self.stop()
        else:
            self._target = self._max
            self._speed = 90.0
            self._rotating = True
            self._opened = True

    def close(self):
        if self._target == self._min:
            self.stop()
        else:
            self._target = self._min
            self._speed = -90.0
            self._rotating = True
            self._opened = False

    def stop(self):
        self._rotating = False
        self._speed = 0

    def _rotate(self, angle):
        self.mesh.rotate_vector(self.axis, angle, point=self.center, inplace=True)
        self.actor.GetMapper().SetInputData(self.mesh)

    def get_menu_actions(self):
        if self._opened:
            return [("🔒 Закрыть", "close")]
        else:
            return [("🔓 Открыть", "open")]

    def execute_action(self, action):
        if action == "open":
            self.open()
        elif action == "close":
            self.close()

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

    def set_pressure_mpa(self, mpa):
        mpa = max(0, min(16, mpa))
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
                ("📊 4 МПа", "set_4"),
                ("📊 8 МПа", "set_8"),
                ("🔧 Снять", "remove"),
            ]
        else:
            return [("🔧 Установить", "attach")]

    def execute_action(self, action):
        if action == "set_4":
            self.set_pressure_mpa(4)
        elif action == "set_8":
            self.set_pressure_mpa(8)
        elif action == "set_12":
            self.set_pressure_mpa(12)
        elif action == "set_16":
            self.set_pressure_mpa(16)
        elif action == "remove":
            self.remove()
        elif action == "attach":
            self.attach()


# ========================
# КОРПУС
# ========================

class Body(Detail):
    def __init__(self, mesh, actor, name, axis=(0, 1, 0), color=None):
        super().__init__(mesh, actor, name, axis)
        self.highlightable = False

class LevelGaugeScreen(Detail):
    def __init__(self, mesh, actor, name, plotter, color=None):
        super().__init__(mesh, actor, name)
        self.plotter = plotter
        self.highlightable = False
        self._screen_plane_mesh = None
        self._screen_plane_actor = None
        self._texture = None
        self._tex_w = 512
        self._tex_h = 512
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
        font_size = max(8, int(line_height * 0.90))

        font = QFont("Consolas", font_size)
        font.setBold(False)
        painter.setFont(font)

        is_header = True

        y = padding_top + line_height
        for line in lines[:max_lines]:
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

    def __init__(self, plotter, position, direction=(0, 1, 0), particle_type=OIL, count=760):
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

        if particle_type == self.OIL:
            self._color = "black"
            self._opacity = 0.64
            self._gravity = np.array([0, -7, 0])
            self._lifetime_min = 1.2
            self._lifetime_max = 1.9
            self._speed_min = 1.9
            self._speed_max = 3.1
        elif particle_type == self.AIR_BLAST:
            self._color = "white"
            self._opacity = 0.25
            self._speed_min = 20.0
            self._speed_max = 30.0
            self._lifetime_min = 0.60
            self._lifetime_max = 1.00
            self._gravity = np.array([0.0, -0.2, 0.0])
        else:
            self._color = "lightgray"
            self._opacity = 0.17
            self._gravity = np.array([0, -2.8, 0])
            self._lifetime_min = 0.1
            self._lifetime_max = 0.8
            self._speed_min = 3.4
            self._speed_max = 5.6

        self._positions = np.tile(self.position, (count, 1))
        self._velocities = np.zeros((count, 3))
        self._lifetimes = np.random.uniform(0, 2.0, count)
        self._mesh = pv.PolyData(self._positions)
        self._actor = plotter.add_mesh(
            self._mesh,
            color=self._color,
            point_size=self._base_point_size,
            render_points_as_spheres=True,
            opacity=self._opacity
        )
        self._actor.VisibilityOff()

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

        self._actor.GetProperty().SetPointSize(new_size)

    def show(self):
        self._actor.VisibilityOn()

    def hide(self):
        self._actor.VisibilityOff()

    def start(self):
        self._active = True
        self._lifetimes = np.random.uniform(0, self._lifetime_max, self._count)
        self._actor.VisibilityOn()

    def stop(self):
        self._active = False
        self._positions[:] = self.position
        self._velocities[:] = 0
        self._lifetimes[:] = 0
        self._mesh.points = self._positions
        self._actor.GetMapper().SetInputData(self._mesh)
        self._actor.VisibilityOff()

    def tick(self, dt=0.016):
        if not self._active:
            return

        for i in range(self._count):
            if self._lifetimes[i] <= 0:
                if self._type == self.AIR_BLAST:
                    angle1 = np.random.uniform(0, np.pi / 120)
                else:
                    angle1 = np.random.uniform(0, np.pi / 6)

                angle2 = np.random.uniform(0, 2 * np.pi)
                dir_rotated = self._rotate_cone(self.direction, angle1, angle2)
                speed = np.random.uniform(self._speed_min, self._speed_max)
                self._positions[i] = self.position.copy()
                self._velocities[i] = dir_rotated * speed
                self._lifetimes[i] = np.random.uniform(self._lifetime_min, self._lifetime_max)
            else:
                self._velocities[i] += self._gravity * dt
                self._positions[i] += self._velocities[i] * dt
                self._lifetimes[i] -= dt

        self._mesh.points = self._positions
        self._actor.GetMapper().SetInputData(self._mesh)
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

    def _rot(self, v, axis, angle):
        axis = axis / np.linalg.norm(axis)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return v * cos_a + np.cross(axis, v) * sin_a + axis * np.dot(axis, v) * (1 - cos_a)