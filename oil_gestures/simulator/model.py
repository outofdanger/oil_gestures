from oil_gestures.simulator.model_loader import load_model
from oil_gestures.simulator.details_and_particles import ParticleSystem, Manometer
from oil_gestures.simulator.level_gauge_ui import LevelGaugeUIState
from oil_gestures.simulator.controller_ui import ControllerUIState
from oil_gestures.simulator.details_and_particles import LevelGaugeAssembly

class Model:
    def __init__(self, plotter, filepath):
        self.plotter = plotter
        self.details = load_model(plotter, filepath)
        self.particle_systems = {}
        self._active = set()
        self._highlighted = None
        self.level_gauge_ui = LevelGaugeUIState()
        self.level_gauge_screen = None
        self.controller_ui = ControllerUIState()
        self.controller_screen = None
        self.level_gauge_assembly = None
        for d in self.details:
            if isinstance(d, LevelGaugeAssembly) and d.name == "level_gauge":
                self.level_gauge_assembly = d
                break
        for d in self.details:
            if isinstance(d, Manometer):
                d.create_gauge(plotter)

        for d in self.details:
            if getattr(d, "name", None) == "level_gauge_screen":
                self.level_gauge_screen = d
                break

        for d in self.details:
            if getattr(d, "name", None) == "controller_screen":
                self.controller_screen = d
                break

        if self.level_gauge_screen is not None:
            self.level_gauge_screen.render_lines(self.level_gauge_ui.get_lines())

        if self.controller_screen is not None:
            self.controller_screen.render_lines(self.controller_ui.get_lines())

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
            particle_type=ParticleSystem.GAS, 
            count=300
        )

        self.particle_systems["5"] = ParticleSystem(
            plotter,
            position=(-4.11, 2.1, -0.11),
            direction=(-1, 0, 0),
            particle_type=ParticleSystem.GAS, 
            count=300
        )

        self.particle_systems["6"] = ParticleSystem(
            plotter,
            position=(-1.82, 1.1, -0.12),
            direction=(-1, 0, 0),
            particle_type=ParticleSystem.GAS, 
            count=300
        )


        CHAINS = {
            "1": {
                "valves": ["valve_4", "valve_5", "valve_12"],
                "blocker": "manometer_2",
            },
            "2": {
                "valves": ["valve_4", "valve_3", "valve_13"],
                "blocker": "manometer_3",
            },
            "3": {
                "valves": ["valve_4", "valve_3", "valve_15"],
                "blocker": "",
            },
            "4": {
                "valves": ["valve_1", "valve_11"],
                "blocker": "manometer_1",
            },
            "5": {
                "valves": ["valve_1"],
                "blocker": "plug",
            },
            "6": {
                "valves": ["valve_14"],
                "blocker": "manometer_4",
            },
        }
        self.chains = CHAINS


        self.MANOMETER_MAX_MPA = 12

        # В __init__ после загрузки:
        self._valve_cache = {}
        for name in ["valve_1", "valve_2", "valve_3", "valve_4", "valve_5", "valve_11", "valve_12", "valve_13", "valve_14", "valve_15"]:
            self._valve_cache[name] = self.get_by_name(name)

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

    def get_menu_actions(self, detail, level_gauge_zoomed=False, controller_zoomed=False, manometer_zoomed=False):
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

        if detail.name in {
            "controller_body",
            "controller_screen",
            "controller_panel"
        }:
            actions = []

            if controller_zoomed:
                actions.append(("🔎 Отдалить", "unfocus_controller"))
            else:
                actions.append(("🔍 Приблизить", "focus_controller"))

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

        if detail.name == "level_gauge_cover":
            return detail.get_menu_actions()

        if isinstance(detail, Manometer):
            actions = []

            if manometer_zoomed:
                actions.append(("🔎 Отдалить", "unfocus_manometer"))
            else:
                actions.append(("🔍 Приблизить", "focus_manometer"))

            actions.extend(detail.get_menu_actions())
            return actions

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

        # Обновить фонтаны по цепочкам вентилей
        for key, config in self.chains.items():
            min_percent = 100
            for vname in config["valves"]:
                valve = self._valve_cache.get(vname)
                if valve:
                    percent = (valve._home / valve._max) * 100
                    min_percent = min(min_percent, percent)
            
            ps = self.particle_systems.get(key)
            m_name = config.get("blocker")
            m = self.get_by_name(m_name) if m_name else None

            if self.level_gauge_assembly and self.level_gauge_assembly.state == "attached":
                for ps in self.particle_systems.values():
                    ps.stop()

            if m and m.state == "attached":
                if hasattr(m, 'set_pressure_mpa'):
                    m.set_pressure_mpa(min_percent * self.MANOMETER_MAX_MPA / 100)
                if ps:
                    ps.stop()
            else:
                if m and hasattr(m, 'set_pressure_mpa'):
                    m.set_pressure_mpa(0)
                if ps:
                    if min_percent > 0:
                        if not ps._active:
                            ps.start()
                        ps.set_intensity(min_percent)
                    else:
                        ps.stop()


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

    def get_controller_bounds(self):
        controller_parts = [
            d for d in self.details
            if getattr(d, "name", "").startswith("controller_")
        ]

        if not controller_parts:
            return None

        xs = []
        ys = []
        zs = []

        for part in controller_parts:
            b = part.bounds
            xs.extend([b[0], b[1]])
            ys.extend([b[2], b[3]])
            zs.extend([b[4], b[5]])

        return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))

    def update_level_gauge_screen(self):
        if self.level_gauge_screen is not None:
            assembly = getattr(self.level_gauge_screen, "parent_assembly", None)
            if assembly and assembly.state == "attached":
                self.level_gauge_screen.render_lines(self.level_gauge_ui.get_lines())

    def update_controller_screen(self):
        if self.controller_screen is not None:
            self.controller_screen.render_lines(self.controller_ui.get_lines())