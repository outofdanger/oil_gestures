import sys
from PySide6.QtWidgets import QApplication

from oil_gestures.ui.main_window import MainWindow


if __name__ == "__main__":
    print("Открытие приложения:")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())