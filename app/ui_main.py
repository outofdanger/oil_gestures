import os
import sys

# Заставляем Qt использовать аппаратный desktop OpenGL, а не ANGLE/софтовый
# рендер. На Windows Qt иногда скатывается в software GL (или ANGLE поверх D3D)
# -> сцена рисуется на CPU, дискретный GPU неактивен. QT_OPENGL читается при
# инициализации Qt, поэтому ставим ДО импорта/создания QApplication.
os.environ.setdefault("QT_OPENGL", "desktop")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from oil_gestures.ui.main_window import MainWindow  # noqa: E402


if __name__ == "__main__":
    print("Открытие приложения:")
    # То же самое, но через атрибут приложения (должен быть выставлен ДО
    # создания QApplication) - дублирует QT_OPENGL для надёжности.
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL, True)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
