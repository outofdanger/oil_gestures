from oil_gestures.simulator.model_loader import load_model
from oil_gestures.simulator.details_and_particles import ParticleSystem, Manometer


class Model:
    def __init__(self, plotter, filepath):
        self.plotter = plotter
        self.details = load_model(plotter, filepath)
        self.particle_systems = {}
        self._active = set()
        self._highlighted = None

        for d in self.details:
            if isinstance(d, Manometer):
                d.create_gauge(plotter)

        # Для заглушки — нефть (чёрная)
        self.particle_systems["1"] = ParticleSystem(
            plotter,
            position=(-0.6, 8.16, -0.11),
            direction=(0, 1, 0),
            particle_type=ParticleSystem.OIL
        )

        self.particle_systems["2"] = ParticleSystem(
            plotter,
            position=(2.14, 6.3, -0.11),
            direction=(0, 1, 0),
            particle_type=ParticleSystem.OIL
        )

        self.particle_systems["3"] = ParticleSystem(
            plotter,
            position=(4.7, 4, -0.11),
            direction=(0, -1, 0),
            particle_type=ParticleSystem.OIL
        )

        self.particle_systems["4"] = ParticleSystem(
            plotter,
            position=(-3.9, 3.5, -0.11),
            direction=(0, 1, 0),
            particle_type=ParticleSystem.GAS
        )

        self.particle_systems["5"] = ParticleSystem(
            plotter,
            position=(-4.11, 2.97, -0.11),
            direction=(-1, 0, 0),
            particle_type=ParticleSystem.GAS
        )

        self.particle_systems["6"] = ParticleSystem(
            plotter,
            position=(-1.9, 2, -0.11),
            direction=(-1, 0, 0),
            particle_type=ParticleSystem.GAS
        )

    # ========================
    #  ПОДСВЕТКА
    # ========================

    def get_by_actor(self, actor):
        for d in self.details:
            if d.actor == actor:
                return d
            
            # Проверяем акторы манометра (циферблат, стрелка)
            if hasattr(d, '_gauge_face') and d._gauge_face == actor:
                return d
            if hasattr(d, '_gauge_arrow') and d._gauge_arrow == actor:
                return d
        
        return None

    def highlight(self, detail):
        if detail == self._highlighted:
            return
        if self._highlighted:
            self._highlighted.unhighlight()
        self._highlighted = detail
        if detail:
            detail.highlight()

    def get_highlighted(self):
        return self._highlighted

    # ========================
    #  МЕНЮ
    # ========================

    def get_menu_actions(self, detail):
        return detail.get_menu_actions()

    def execute_action(self, detail, action):
        detail.execute_action(action)
        if detail.has_animation():
            self._active.add(detail)


    # ========================
    #  ТАЙМЕР
    # ========================

    def has_active(self):
        if self._active:
            return True
        for ps in self.particle_systems.values():
            if ps._active:
                return True
        return False

    def tick(self, dt=0.016):
        done = []
        for d in self._active:
            d.tick_animation(dt)
            if not d.has_animation():
                done.append(d)
        for d in done:
            self._active.discard(d)

        for ps in self.particle_systems.values():
            if ps._active:
                ps.tick(dt)

    # ========================
    #  ИНВЕНТАРЬ
    # ========================

    def get_inventory(self):
        items = []
        for d in self.details:
            if hasattr(d, 'state') and d.state == "removed":
                items.append(d.name)
        return items


    # ========================
    #  АВАРИЙНЫЙ СТОП
    # ========================

    def emergency_stop(self):
        """Останавливает все потоки нефти/газа. FIST -> EMERGENCY_STOP, см. docs/command_mapping.md."""
        for ps in self.particle_systems.values():
            ps.stop()

    def get_by_name(self, name):
        for d in self.details:
            if d.name == name:
                return d
        return None