class ControllerUIState:
    def __init__(self):
        self.power_on = False
        self.running = False
        self.status = "ПИТАНИЕ ВЫКЛ"

    def toggle_power(self):
        self.power_on = not self.power_on

        if self.power_on:
            self.status = "ПИТАНИЕ ВКЛ"
        else:
            self.running = False
            self.status = "ПИТАНИЕ ВЫКЛ"

    def press_start(self):
        if not self.power_on:
            self.status = "НЕТ ПИТАНИЯ"
            return False

        self.running = True
        self.status = "УСТАНОВКА ЗАПУЩЕНА"
        return True

    def press_stop(self):
        if not self.power_on:
            self.status = "НЕТ ПИТАНИЯ"
            return False

        self.running = False
        self.status = "УСТАНОВКА ОСТАНОВЛЕНА"
        return True

    def get_lines(self):
        power = "ВКЛ" if self.power_on else "ВЫКЛ"
        running = "ДА" if self.running else "НЕТ"

        return [
            "КОНТРОЛЛЕР",
            f"ПИТАНИЕ: {power}",
            f"РАБОТА: {running}",
            "",
            self.status,
        ]