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
            "Давление: 8.2 МПа",
            "Статус: норма",
        ]
        self.result_index = 0
        self._level_measured = False
        self._pressure_measured = False

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
            if self._level_measured:
                lines.append("1200 м")
            else:
                lines.append("--")
            return lines

        if self.current_screen == "measure_pressure":
            lines = ["ИЗМЕРЕНИЕ ДАВЛЕНИЯ", "", "Текущее давление:"]
            if self._pressure_measured:
                lines.append("8.2 МПа")
            else:
                lines.append("--")
            return lines

        if self.current_screen == "results":
            all_results = [
                "Уровень: 1200 м" if self._level_measured else "Уровень: --",
                "Давление: 8.2 МПа" if self._pressure_measured else "Давление: --",
                "Статус: норма",
            ]
            return [
                "РЕЗУЛЬТАТЫ",
                f"{self.result_index + 1}/{len(all_results)}",
                "",
                all_results[self.result_index],
            ]

        return ["УРОВНЕМЕР"]