import numpy as np


class Camera:
    """Камера сцены: вращение, zoom, перемещение, ракурсы."""

    def __init__(self, plotter):
        self.plotter = plotter
        self._default = None
        self._min_zoom = 2.4
        self._max_zoom = 73.0

        self._speed = 0
        self._decaying = False

        self._saved_position = None

    def save_default(self):
        self._default = self._copy_camera_position(self.plotter.camera_position)

    def reset(self):
        if self._default:
            self.plotter.camera_position = self._copy_camera_position(self._default)

    def _copy_camera_position(self, cam_pos):
        pos, focus, viewup = cam_pos
        return [tuple(pos), tuple(focus), tuple(viewup)]

    # ========================
    # ВРАЩЕНИЕ
    # ========================

    def start_rotate(self, speed):
        self._speed = speed
        self._decaying = False

    def stop_rotate(self):
        self._decaying = True

    def is_rotating(self):
        return abs(self._speed) > 0.001

    def tick(self, dt=0.016):
        if self._decaying:
            self._speed *= 0.36
        if abs(self._speed) < 0.001:
            self._speed = 0
            return
        self._apply_rotation(self._speed * dt)

    def _apply_rotation(self, angle_deg):
        pos, focus, viewup = self.plotter.camera_position
        angle = np.radians(angle_deg)
        direction = np.array(pos) - np.array(focus)
        axis = np.array((0, 1, 0))
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotated = (
            direction * cos_a
            + np.cross(axis, direction) * sin_a
            + axis * np.dot(axis, direction) * (1 - cos_a)
        )
        new_pos = tuple(np.array(focus) + rotated)
        self.plotter.camera_position = [new_pos, focus, (0, 1, 0)]

    # ========================
    # ZOOM
    # ========================

    def zoom(self, factor):
        pos, focus, viewup = self.plotter.camera_position
        direction = np.array(focus) - np.array(pos)
        dist = np.linalg.norm(direction)
        new_dist = max(self._min_zoom, min(self._max_zoom, dist * factor))
        direction = direction / dist
        new_pos = tuple(np.array(focus) - direction * new_dist)
        self.plotter.camera_position = [new_pos, focus, (0, 1, 0)]

    # ========================
    # ФОКУС НА УРОВНЕМЕР
    # ========================

    def save_current_view(self):
        self._saved_position = self._copy_camera_position(self.plotter.camera_position)

    def restore_saved_view(self):
        if self._saved_position is not None:
            self.plotter.camera_position = self._copy_camera_position(self._saved_position)
            self._saved_position = None

    def focus_on_level_gauge(self, bounds):
        x0, x1, y0, y1, z0, z1 = bounds

        sx = x1 - x0
        sy = y1 - y0
        sz = z1 - z0
        size = max(sx, sy, sz, 0.5)

        focus = np.array([
            x0 + sx * 0.50,
            y0 + sy * 0.55,
            z0 + sz * 0.55,
        ], dtype=float)

        distance = max(self._min_zoom, min(self._max_zoom, size * 2.6))

        pos = np.array([
            focus[0],
            focus[1] + size * 0.08,
            z1 + distance,
        ], dtype=float)

        self.plotter.camera_position = [tuple(pos), tuple(focus), (0, 1, 0)]

    # ========================
    # ПЕРЕМЕЩЕНИЕ
    # ========================

    def move(self, dx=0, dy=0, dz=0):
        pos, focus, viewup = self.plotter.camera_position
        if dx != 0:
            direction = np.array(focus) - np.array(pos)
            forward = np.array([direction[0], 0, direction[2]])
            if np.linalg.norm(forward) > 0.001:
                forward = forward / np.linalg.norm(forward)
            else:
                forward = np.array([0, 0, 1])
            right = np.array([-forward[2], 0, forward[0]])
            shift = right * dx
            new_pos = (pos[0] + shift[0], pos[1] + dy, pos[2] + shift[2])
            new_focus = (focus[0] + shift[0], focus[1] + dy, focus[2] + shift[2])
        else:
            new_pos = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
            new_focus = (focus[0] + dx, focus[1] + dy, focus[2] + dz)
        self.plotter.camera_position = [new_pos, new_focus, (0, 1, 0)]

    # ========================
    # РАКУРСЫ
    # ========================

    def view_top(self):
        self.plotter.view_xy()

    def view_front(self):
        self.plotter.view_xz()

    def view_side(self):
        self.plotter.view_yz()

    def view_iso(self):
        self.plotter.view_isometric()