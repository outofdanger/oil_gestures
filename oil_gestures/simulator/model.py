from oil_gestures.simulator.model_loader import load_model
from oil_gestures.simulator.details_and_particles import ParticleSystem, Manometer
from oil_gestures.simulator.level_gauge_ui import LevelGaugeUIState

class Model:
    def __init__(self, plotter, filepath):
        self.plotter = plotter
        self.details = load_model(plotter, filepath)
        self.particle_systems = {}
        self._active = set()
        self._highlighted = None
        self.level_gauge_ui = LevelGaugeUIState()
        self.level_gauge_screen = None

        for d in self.details:
            if isinstance(d, Manometer):
                d.create_gauge(plotter)

        for d in self.details:
            if getattr(d, "name", None) == "level_gauge_screen":
                self.level_gauge_screen = d
                break

        if self.level_gauge_screen is not None:
            self.level_gauge_screen.render_lines(self.level_gauge_ui.get_lines())

        self.particle_systems["1"] = ParticleSystem(
            plotter,
            position=(-0.6, 7, -0.11),
            direction=(0, 1, 0),
            particle_type=ParticleSystem.OIL
        )

        self.particle_systems["2"] = ParticleSystem(
            plotter,
            position=(2.17, 5.54, -0.12),
            direction=(0, 1, 0),
            particle_type=ParticleSystem.OIL
        )

        self.particle_systems["3"] = ParticleSystem(
            plotter,
            position=(4.66, 3.02, -0.11),
            direction=(0, -1, 0),
            particle_type=ParticleSystem.OIL
        )

        self.particle_systems["4"] = ParticleSystem(
            plotter,
            position=(-3.93, 2.69, -0.11),
            direction=(0, 1, 0),
            particle_type=ParticleSystem.GAS
        )

        self.particle_systems["5"] = ParticleSystem(
            plotter,
            position=(-4.11, 2.1, -0.11),
            direction=(-1, 0, 0),
            particle_type=ParticleSystem.GAS
        )

        self.particle_systems["6"] = ParticleSystem(
            plotter,
            position=(-1.82, 1.1, -0.12),
            direction=(-1, 0, 0),
            particle_type=ParticleSystem.GAS
        )

    def get_by_actor(self, actor):
        for d in self.details:
            if getattr(d, "actor", None) == actor:
                return d
            if hasattr(d, "_gauge_face") and d._gauge_face == actor:
                return d
            if hasattr(d, "_gauge_arrow") and d._gauge_arrow == actor:
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

    def get_menu_actions(self, detail, level_gauge_zoomed=False):
        if detail.name == "level_gauge_base":
            actions = []

            if level_gauge_zoomed:
                actions.append(("🔎 Отдалить", "unfocus_level_gauge"))
            else:
                actions.append(("🔍 Приблизить", "focus_level_gauge"))

            assembly = getattr(detail, "parent_assembly", None)
            if assembly and getattr(assembly, "state", None) == "attached":
                actions.append(("🔧 Снять", "remove_level_gauge"))

            return actions

        BUTTON_INFO = {
            "level_gauge_button_mode": [
                ("ℹ РЕЖИМ", None),
                ("• Измерение уровня", None),
                ("• Измерение давления", None),
                ("• Просмотр результатов", None),
            ],
            "level_gauge_button_input_output": [
                ("ℹ ВВОД/ВЫВОД", None),
                ("• Подтвердить выбранный режим", None),
                ("• В режиме просмотра — листать результаты", None),
            ],
            "level_gauge_button_level": [
                ("ℹ УРОВЕНЬ", None),
                ("• Сразу включает режим 'Измерение уровня'", None),
            ],
            "level_gauge_button_return": [
                ("ℹ ВОЗВРАТ", None),
                ("• Возвращает на главный экран", None),
            ],
        }

        if detail.name in BUTTON_INFO:
            return BUTTON_INFO[detail.name]

        if getattr(detail, "parent_assembly", None) is not None:
            return []

        return detail.get_menu_actions()

    def execute_action(self, detail, action):
        if action == "remove_level_gauge":
            assembly = getattr(detail, "parent_assembly", None)
            if assembly:
                assembly.remove()
            return

        detail.execute_action(action)
        if detail.has_animation():
            self._active.add(detail)

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

    def get_inventory(self):
        items = []
        for d in self.details:
            if hasattr(d, "state") and d.state == "removed":
                items.append(d.name)
        return items

    def get_by_name(self, name):
        for d in self.details:
            if d.name == name:
                return d
        return None

    def update_level_gauge_screen(self):
        if self.level_gauge_screen is not None:
            assembly = getattr(self.level_gauge_screen, "parent_assembly", None)
            if assembly and assembly.state == "attached":
                self.level_gauge_screen.render_lines(self.level_gauge_ui.get_lines())