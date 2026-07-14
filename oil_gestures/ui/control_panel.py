from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt, Signal


class ControlPanel(QWidget):
    """Правая панель управления."""

    help_clicked = Signal()
    emergency_clicked = Signal()
    inventory_item_clicked = Signal(str) 

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(230)
        self.setStyleSheet("background-color: gray;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 15, 12, 15)
        layout.setSpacing(10)

        # Заголовок
        title = QLabel("ПАНЕЛЬ УПРАВЛЕНИЯ")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

                # Состояние модели (крупно, как сейчас жесты)
        self.model_state = QLabel("СОСТОЯНИЕ МОДЕЛИ\nОжидание")
        self.model_state.setAlignment(Qt.AlignCenter)
        self.model_state.setWordWrap(True)
        self.model_state.setStyleSheet(
            "color: #2ecc71; font-size: 12px; font-weight: bold; "
            "background-color: #1a1a2e; padding: 7px; border-radius: 6px;"
        )
        layout.addWidget(self.model_state)

        # Виджет камеры
        self.camera_label = QLabel("КАМЕРА")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumHeight(150)
        self.camera_label.setStyleSheet(
            "background-color: white; color: gray; border: 2px solid #34495e;"
        )
        layout.addWidget(self.camera_label)

        # Состояние жеста (поменьше)
        self.gesture_state = QLabel("ОБРАБОТКА ЖЕСТОВ\n—")
        self.gesture_state.setAlignment(Qt.AlignCenter)
        self.gesture_state.setStyleSheet(
            "color: #3498db; font-size: 11px; font-weight: bold; "
            "background-color: #1a1a2e; padding: 6px; border-radius: 6px;"
        )
        layout.addWidget(self.gesture_state)

        # Инвентарь
        self.inventory_label = QLabel("ИНВЕНТАРЬ")
        self.inventory_label.setStyleSheet("color: #f39c12; font-size: 14px; font-weight: bold; padding: 4px;")
        layout.addWidget(self.inventory_label)

        self.inventory_layout = QVBoxLayout()
        self.inventory_layout.setSpacing(4)
        layout.addLayout(self.inventory_layout)

        layout.addStretch()

        # Кнопка справки
        self.help_btn = QPushButton("СПРАВКА")
        self.help_btn.setMinimumHeight(40)
        self.help_btn.setStyleSheet("""
            QPushButton {
                background-color: #2980b9; color: white; font-size: 14px;
                font-weight: bold; border: none; border-radius: 6px;
            }
            QPushButton:hover { background-color: #3498db; }
            QPushButton:pressed { background-color: #1c6ea4; }
        """)
        self.help_btn.clicked.connect(self.help_clicked.emit)
        layout.addWidget(self.help_btn)

        # Аварийный стоп
        self.emergency_btn = QPushButton("АВАРИЙНЫЙ СТОП")
        self.emergency_btn.setMinimumHeight(50)
        self.emergency_btn.setStyleSheet("""
            QPushButton {
                background-color: #c0392b; color: white; font-size: 16px;
                font-weight: bold; border: none; border-radius: 6px;
            }
            QPushButton:hover { background-color: #e74c3c; }
            QPushButton:pressed { background-color: #922b21; }
        """)
        self.emergency_btn.clicked.connect(self.emergency_clicked.emit)
        layout.addWidget(self.emergency_btn)

    def set_gesture(self, text: str):
        self.gesture_state.setText(text)

    def set_model_state(self, text: str):
        self.model_state.setText(text)

    def set_inventory(self, items: list):
        # Очистить старые кнопки
        while self.inventory_layout.count():
            child = self.inventory_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if not items:
            empty = QLabel("   пусто")
            empty.setStyleSheet("color: #7f8c8d; font-size: 12px;")
            self.inventory_layout.addWidget(empty)
            return

        for item in items:
            # Элемент инвентаря - кортеж (внутреннее имя, отображаемое имя).
            # Для обратной совместимости поддержим и случай, если придёт просто строка.
            if isinstance(item, (tuple, list)):
                name, display_name = item[0], item[1]
            else:
                name, display_name = item, item

            btn = QPushButton(f"📦 {display_name}")
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #34495e; color: white; font-size: 12px;
                    padding: 6px; border-radius: 4px; text-align: left;
                }
                QPushButton:hover { background-color: #2980b9; }
            """)
            btn.clicked.connect(lambda checked, name=name: self.inventory_item_clicked.emit(name))
            self.inventory_layout.addWidget(btn)

    def set_camera_image(self, image):
        """Показать уже декодированный и отмасштабированный кадр камеры."""
        from PySide6.QtGui import QPixmap
        if image is None or image.isNull():
            return
        self.camera_label.setPixmap(QPixmap.fromImage(image))