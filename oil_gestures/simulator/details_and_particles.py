import numpy as np
import pyvista as pv


# ========================
#  БАЗОВАЯ ДЕТАЛЬ
# ========================

class Detail:
    def __init__(self, mesh, actor, name, color=None):
        self.mesh = mesh
        self.actor = actor
        self.name = name
        self._original_color = actor.GetProperty().GetColor()
        self.highlightable = True
        self._highlight_color = (0.0, 1.0, 0.0)


    def highlight(self):
        if self.highlightable:
            self.actor.GetProperty().SetColor(*self._highlight_color)

    def unhighlight(self):
        if self.highlightable:
            self.actor.GetProperty().SetColor(*self._original_color)

    # === Видимость (общая) ===

    def show(self):
        self.actor.VisibilityOn()

    def hide(self):
        self.actor.VisibilityOff()

    # === Меню (потомки переопределяют) ===

    def get_menu_actions(self):
        return []

    def execute_action(self, action):
        pass

    # === Анимация (потомки переопределяют) ===

    def has_animation(self):
        return False

    def tick_animation(self, dt):
        pass

    # === Свойства (общие) ===

    @property
    def center(self):
        return self.mesh.center

    @property
    def bounds(self):
        return self.mesh.bounds


# ========================
#  ВЕНТИЛЬ
# ========================

class Valve(Detail):
    def __init__(self, mesh, actor, name, color=None):
        super().__init__(mesh, actor, name)
        self._rotating = False
        self._speed = 0
        self._target = 0
        self._home = 0
        self._min = 0
        self._max = 360
        self._opened = False

    @property
    def is_open(self):
        return self._opened

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
        self.mesh.rotate_vector((0, 0, 1), angle, point=self.center, inplace=True)
        self.actor.GetMapper().SetInputData(self.mesh)

    def get_menu_actions(self):
        if self._opened:
            return [("🔒 Закрыть", "close")]
        else:
            return [("🔓 Открыть", "open")]

    def execute_action(self, action):
        if action == "open": self.open()
        elif action == "close": self.close()

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

# ========================
#  ЗАГЛУШКА
# ========================

class Plug(Detail):
    def __init__(self, mesh, actor, name, color=None):
        super().__init__(mesh, actor, name)
        self.state = "attached"

    def remove(self):
        self.state = "removed"
        self.hide()

    def attach(self):
        self.state = "attached"
        self.show()

    # === Меню ===

    def get_menu_actions(self):
        if self.state == "attached":
            return [("🔧 Снять", "remove")]
        else:
            return [("🔧 Установить", "attach")]

    def execute_action(self, action):
        if action == "remove": self.remove()
        elif action == "attach": self.attach()


# ========================
#  МАНОМЕТР
# ========================

class Manometer(Detail):
    def __init__(self, mesh, actor, name, color=None):
        super().__init__(mesh, actor, name)
        self._gauge_face = None
        self._gauge_arrow = None
        self._arrow_mesh = None
        self._arrow_center = None
        self._home_angle = 0
        self.state = "attached"

    def create_gauge(self, plotter):
        """Создать циферблат и стрелку."""
        center = self.mesh.center
        pos = (center[0], center[1], center[2] + 0.039)
        size = 0.31

        # Циферблат
        plane = pv.Plane(center=pos, direction=(0, 0, 1), i_size=size, j_size=size)
        try:
            tex = pv.read_texture("assets/ui/gauge_face.png")
            self._gauge_face = plotter.add_mesh(plane, texture=tex, opacity=1.0)
        except (FileNotFoundError, OSError):
            self._gauge_face = plotter.add_mesh(plane, color="white", opacity=0.9)

        # Привязываем актор шкалы к манометру для подсветки
        self._gauge_actor = self._gauge_face

        # Стрелка
        line = pv.Line(
            (pos[0] + 0.003, pos[1], pos[2] + 0.005),
            (pos[0] + size * 0.43, pos[1], pos[2] + 0.005)
        )
        arrow = line.tube(radius=0.0028)
        tip = pv.Sphere(radius=0.011, center=(pos[0], pos[1], pos[2]))
        arrow = arrow.merge(tip)
        self._gauge_arrow = plotter.add_mesh(arrow, color="red")
        self._arrow_mesh = arrow
        self._arrow_center = (pos[0], pos[1], pos[2] + 0.005)

    def set_pressure_mpa(self, mpa):
        """Установить давление в МПа (0..16)."""
        mpa = max(0, min(16, mpa))
        percent = mpa / 16 * 100
        angle = 210 - 240 * (percent / 100)
        delta = angle - self._home_angle
        if self._arrow_mesh:
            self._arrow_mesh.rotate_vector((0, 0, 1), delta, point=self._arrow_center, inplace=True)
            self._gauge_arrow.GetMapper().SetInputData(self._arrow_mesh)
        self._home_angle = angle

    # === Подсветка вместе со шкалой ===

    def highlight(self):
        super().highlight()
        if self._gauge_face:
            self._gauge_face.GetProperty().SetColor(1.0, 1.0, 0.0)

    def unhighlight(self):
        super().unhighlight()
        if self._gauge_face:
            self._gauge_face.GetProperty().SetColor(1.0, 1.0, 1.0)

    # === Меню ===

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
        if action == "set_4": self.set_pressure_mpa(4)
        elif action == "set_8": self.set_pressure_mpa(8)
        elif action == "set_12": self.set_pressure_mpa(12)
        elif action == "set_16": self.set_pressure_mpa(16)
        elif action == "remove": self.remove()
        elif action == "attach": self.attach()

# ========================
#  КОРПУС
# ========================

class Body(Detail):
    def __init__(self, mesh, actor, name, color=None):
        super().__init__(mesh, actor, name)
        self.highlightable = False



# ========================
#  ЧАСТИЦЫ
# ========================

class ParticleSystem:
    """Система частиц: нефть или газ."""

    OIL = "oil"
    GAS = "gas"

    def __init__(self, plotter, position, direction=(0, 1, 0), particle_type=OIL, count=760):
        self.plotter = plotter
        self.position = np.array(position, dtype=float)
        self.direction = np.array(direction, dtype=float)
        self.direction = self.direction / np.linalg.norm(self.direction)
        self._type = particle_type
        self._count = count
        self._active = False

        # Настройки по типу
        if particle_type == self.OIL:
            self._color = "black"
            self._opacity = 0.64
            self._gravity = np.array([0, -7, 0])  # тяжелее
            self._lifetime_min = 1.2
            self._lifetime_max = 1.9
            self._speed_min = 1.9
            self._speed_max = 3.1
        else:  # GAS
            self._color = "lightgray"
            self._opacity = 0.17
            self._gravity = np.array([0, -2.8, 0])   # легче
            self._lifetime_min = 0.1
            self._lifetime_max = 0.8
            self._speed_min = 3.4
            self._speed_max = 5.6

        self._positions = np.tile(self.position, (count, 1))
        self._velocities = np.zeros((count, 3))
        self._lifetimes = np.random.uniform(0, 2.0, count)
        self._mesh = pv.PolyData(self._positions)
        self._actor = plotter.add_mesh(
            self._mesh, color=self._color, point_size=5,
            render_points_as_spheres=True, opacity=self._opacity
        )

    def start(self):
        self._active = True
        self._lifetimes = np.random.uniform(0, self._lifetime_max, self._count)

    def stop(self):
        self._active = False
        # Все частицы исчезают
        self._positions[:] = self.position
        self._velocities[:] = 0
        self._lifetimes[:] = 0
        self._mesh.points = self._positions
        self._actor.GetMapper().SetInputData(self._mesh)

    def tick(self, dt=0.016):
        if not self._active:
            return

        # Полностью векторно (без Python-цикла по частицам): «мёртвые» частицы
        # перерождаются, живые — интегрируются. Это снимает главный CPU-боттлнек.
        dead = self._lifetimes <= 0
        n_dead = int(np.count_nonzero(dead))
        if n_dead:
            dirs = self._random_cone_dirs(n_dead)
            speeds = np.random.uniform(self._speed_min, self._speed_max, (n_dead, 1))
            self._positions[dead] = self.position
            self._velocities[dead] = dirs * speeds
            self._lifetimes[dead] = np.random.uniform(
                self._lifetime_min, self._lifetime_max, n_dead
            )

        alive = ~dead
        if np.any(alive):
            self._velocities[alive] += self._gravity * dt
            self._positions[alive] += self._velocities[alive] * dt
            self._lifetimes[alive] -= dt

        self._mesh.points = self._positions
        self._actor.GetMapper().SetInputData(self._mesh)

    def _random_cone_dirs(self, n):
        """n случайных единичных направлений в конусе вокруг self.direction."""
        v = self.direction
        if abs(v[0]) < 1e-3 and abs(v[2]) < 1e-3:
            perp = np.array([1.0, 0.0, 0.0])
        else:
            perp = np.array([-v[2], 0.0, v[0]])
        perp = perp / np.linalg.norm(perp)

        a1 = np.random.uniform(0, np.pi / 6, n)   # отклонение от оси
        a2 = np.random.uniform(0, 2 * np.pi, n)   # разворот вокруг оси

        base = np.broadcast_to(v, (n, 3))
        tilted = self._rodrigues(base, perp, a1)
        return self._rodrigues(tilted, v, a2)

    @staticmethod
    def _rodrigues(vec, axis, angle):
        """Поворот строк vec вокруг axis на углы angle (формула Родрига, векторно)."""
        axis = axis / np.linalg.norm(axis)
        cos_a = np.cos(angle)[:, None]
        sin_a = np.sin(angle)[:, None]
        cross = np.cross(np.broadcast_to(axis, vec.shape), vec)
        dot = vec @ axis
        return vec * cos_a + cross * sin_a + axis[None, :] * dot[:, None] * (1.0 - cos_a)