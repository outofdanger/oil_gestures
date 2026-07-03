class ControllerUIState:
    MODE_DATA = "ПОЛУЧИТЬ ДАННЫЕ"
    MODE_INCREASE_FLOW = "ПОВЫСИТЬ ДЕБИТ"
    MODE_DECREASE_FLOW = "ПОНИЗИТЬ ДЕБИТ"

    def __init__(self):
        self.power_on = False
        self.running = False
        self.status = "ПИТАНИЕ ВЫКЛ"
        self.current_screen = "home"
        self.modes = [
            self.MODE_DATA,
            self.MODE_INCREASE_FLOW,
            self.MODE_DECREASE_FLOW,
        ]
        self.selected_mode_index = 0

        self.current_amp = 21
        self.voltage = 380
        self.pressure = 8.2
        self.frequency = 50
        self.cosfi = 0.85

    def toggle_power(self):
        self.power_on = not self.power_on

        if self.power_on:
            self.current_screen = "menu"
            self.status = "ВЫБЕРИТЕ РЕЖИМ"
        else:
            self.running = False
            self.current_screen = "home"
            self.status = "ПИТАНИЕ ВЫКЛ"

    def press_start(self):
        if not self._require_power():
            return False

        self.running = True
        self.current_screen = "menu"
        self.status = "УСТАНОВКА ЗАПУЩЕНА"
        return True

    def press_stop(self):
        if not self._require_power():
            return False

        self.running = False
        self.current_screen = "home"
        self.status = "УСТАНОВКА ОСТАНОВЛЕНА"
        return True

    def press_menu(self):
        if not self._require_power():
            return False

        self.current_screen = "menu"
        self.status = "ВЫБЕРИТЕ РЕЖИМ"
        return True

    def press_next(self):
        if not self._require_power():
            return False

        self.current_screen = "menu"
        self.selected_mode_index = (self.selected_mode_index + 1) % len(self.modes)
        self.status = self.selected_mode
        return True

    def press_prev(self):
        if not self._require_power():
            return False

        self.current_screen = "menu"
        self.selected_mode_index = (self.selected_mode_index - 1) % len(self.modes)
        self.status = self.selected_mode
        return True

    def press_confirm(self):
        if not self._require_power():
            return False

        if self.selected_mode == self.MODE_DATA:
            return self.read_data()

        if self.selected_mode == self.MODE_INCREASE_FLOW:
            return self.increase_flow()

        if self.selected_mode == self.MODE_DECREASE_FLOW:
            return self.decrease_flow()

        return False

    def press_back(self):
        if not self._require_power():
            return False

        self.current_screen = "menu"
        self.status = "ВЫБЕРИТЕ РЕЖИМ"
        return True

    @property
    def selected_mode(self):
        return self.modes[self.selected_mode_index]

    def get_lines(self):
        if not self.power_on:
            return [
                "КОНТРОЛЛЕР",
                "",
                "ПИТАНИЕ: ВЫКЛ",
                "",
                "",
                self.status,
            ]

        if self.current_screen == "menu":
            return self._menu_lines()

        if self.current_screen == "data":
            return [
                "ДАННЫЕ",
                f"ТОК: {self.current_amp} А",
                f"НАПРЯЖЕНИЕ: {self.voltage} В",
                f"ДАВЛЕНИЕ: {self.pressure:.1f} МПа",
                f"ЧАСТОТА: {self.frequency} Гц",
                f"COSFI: {self.cosfi:.2f}",
            ]

        running = "RUN" if self.running else "STOP/OFF"
        return [
            "КОНТРОЛЛЕР",
            "ПИТАНИЕ: ВКЛ",
            f"СОСТОЯНИЕ: {running}",
            "",
            "",
            self.status,
        ]

    def read_data(self):
        if not self._require_power():
            return False

        self.current_screen = "data"
        self.status = "ДАННЫЕ ОБНОВЛЕНЫ"
        return True

    def increase_flow(self):
        if not self._require_power():
            return False

        self.current_amp = min(60, self.current_amp + 1)
        self.frequency = min(60, self.frequency + 1)
        self.pressure = min(12.0, round(self.pressure + 0.2, 1))
        self.current_screen = "data"
        self.status = "ДЕБИТ ПОВЫШЕН"
        return True

    def decrease_flow(self):
        if not self._require_power():
            return False

        self.current_amp = max(0, self.current_amp - 1)
        self.frequency = max(0, self.frequency - 1)
        self.pressure = max(0.0, round(self.pressure - 0.2, 1))
        self.current_screen = "data"
        self.status = "ДЕБИТ ПОНИЖЕН"
        return True

    def _menu_lines(self):
        lines = ["ВЫБОР РЕЖИМА"]
        for index, mode in enumerate(self.modes):
            prefix = ">" if index == self.selected_mode_index else " "
            lines.append(f"{prefix} {index + 1}. {mode}")

        lines.extend(["", self.status])
        return lines[:6]

    def _require_power(self):
        if self.power_on:
            return True

        self.current_screen = "home"
        self.status = "НЕТ ПИТАНИЯ"
        return False
