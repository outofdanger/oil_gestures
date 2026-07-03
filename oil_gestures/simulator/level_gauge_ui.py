import random

class LevelGaugeUIState:
    MODE_LEVEL = "Измерение уровня"
    MODE_PRESSURE = "Измерение давления"
    MODE_RESULTS = "Просмотр результатов"

    def __init__(self):
        self.modes = [
            self.MODE_LEVEL,
            self.MODE_PRESSURE,
            self.MODE_RESULTS,
        ]
        self.selected_mode_index = 0
        self.current_screen = "home"
        self.results = [
            "Уровень: 1200 м",
            "Давление: --",
            "Статус: норма",
        ]
        self.result_index = 0
        self._level_measured = False
        self._pressure_measured = False
        self._last_level_m = None  # сохранённое значение последнего измерения
        self._last_pressure_mpa = 0.0

    def press_mode(self):
        if self.current_screen == "home":
            self.current_screen = "mode_select"
            return
        if self.current_screen != "mode_select":
            self.current_screen = "mode_select"
            return
        self.selected_mode_index = (self.selected_mode_index + 1) % len(self.modes)

    def press_input_output(self):
        if self.current_screen == "mode_select":
            mode = self.modes[self.selected_mode_index]
            if mode == self.MODE_LEVEL:
                self.current_screen = "measure_level"
            elif mode == self.MODE_PRESSURE:
                self.current_screen = "measure_pressure"
            else:
                self.current_screen = "results"
                self.result_index = 0
            return

        if self.current_screen == "measure_level":
            return

        if self.current_screen == "measure_pressure":
            return

        if self.current_screen == "results":
            self.result_index = (self.result_index + 1) % len(self.results)

    def press_level(self):
        self.current_screen = "measure_level"

    def complete_level_measurement(self):
        """Вызывается по таймеру когда измерение завершено — генерирует случайное значение."""
        self._last_level_m = random.randint(1000, 1200)
        self._level_measured = True

    def set_pressure_mpa(self, pressure):
        pressure = max(0.0, min(12.0, round(pressure, 1)))
        if abs(self._last_pressure_mpa - pressure) < 0.05:
            return False
        self._last_pressure_mpa = pressure
        return True

    def press_return(self):
        self.current_screen = "home"

    def get_lines(self):
        if self.current_screen == "home":
            return [
                "УРОВНЕМЕР",
                "",
                "ВЫБОР РЕЖИМА",
            ]

        if self.current_screen == "mode_select":
            lines = ["ВЫБОР РЕЖИМА", ""]
            for i, mode in enumerate(self.modes):
                prefix = ">" if i == self.selected_mode_index else " "
                lines.append(f"{prefix} {mode}")
            return lines

        if self.current_screen == "measure_level":
            lines = ["ИЗМЕРЕНИЕ УРОВНЯ", "", "Текущий уровень:"]
            if self._level_measured and self._last_level_m is not None:
                lines.append(f"{self._last_level_m} м")
            else:
                lines.append("--")
            return lines

        if self.current_screen == "measure_pressure":
            lines = ["ИЗМЕРЕНИЕ ДАВЛЕНИЯ", "", "Текущее давление:"]
            if self._pressure_measured:
                lines.append(f"{self._last_pressure_mpa:.1f} МПа")
            else:
                lines.append("--")
            return lines

        if self.current_screen == "results":
            level_str = f"{self._last_level_m} м" if self._level_measured and self._last_level_m is not None else "--"
            all_results = [
                f"Уровень: {level_str}" if self._level_measured else "Уровень: --",
                f"Давление: {self._last_pressure_mpa:.1f} МПа" if self._pressure_measured else "Давление: --",
                "Статус: норма",
            ]
            return [
                "РЕЗУЛЬТАТЫ",
                f"{self.result_index + 1}/{len(all_results)}",
                "",
                all_results[self.result_index],
            ]

        return ["УРОВНЕМЕР"]