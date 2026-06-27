from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt, Signal


class ControlPanel(QWidget):
    """Правая панель управления."""

    emergency_clicked = Signal()
    inventory_item_clicked = Signal(str)  # имя детали

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.setStyleSheet("background-color: #2c3e50;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Заголовок
        title = QLabel("ПАНЕЛЬ УПРАВЛЕНИЯ")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        # Текущий статический жест - отдельный, всегда актуальный индикатор,
        # независимый от лога действий ниже (message).
        gesture_caption = QLabel("ТЕКУЩИЙ ЖЕСТ")
        gesture_caption.setAlignment(Qt.AlignCenter)
        gesture_caption.setStyleSheet("color: #7f8c8d; font-size: 11px; font-weight: bold;")
        layout.addWidget(gesture_caption)

        self.gesture_value = QLabel("—")
        self.gesture_value.setAlignment(Qt.AlignCenter)
        self.gesture_value.setStyleSheet(
            "color: #2ecc71; font-size: 16px; font-weight: bold; "
            "background-color: #1a1a2e; padding: 8px; border-radius: 6px;"
        )
        layout.addWidget(self.gesture_value)

        # Сообщение
        self.message = QLabel("Ожидание действий...")
        self.message.setAlignment(Qt.AlignCenter)
        self.message.setWordWrap(True)
        self.message.setStyleSheet(
            "color: #bdc3c7; font-size: 13px; background-color: #1a1a2e; "
            "padding: 12px; border-radius: 6px;"
        )
        layout.addWidget(self.message)


        # В __init__ после message:
        self.camera_label = QLabel("КАМЕРА")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumHeight(150)
        self.camera_label.setStyleSheet(
            "background-color: #1a1a2e; color: gray; border: 1px solid #34495e; border-radius: 4px;"
        )
        layout.addWidget(self.camera_label)



        # Инвентарь
        self.inventory_label = QLabel("ИНВЕНТАРЬ")
        self.inventory_label.setStyleSheet("color: #f39c12; font-size: 12px; font-weight: bold; padding: 4px;")
        layout.addWidget(self.inventory_label)

        self.inventory_layout = QVBoxLayout()
        self.inventory_layout.setSpacing(4)
        layout.addLayout(self.inventory_layout)

        layout.addStretch()

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

    def set_message(self, text: str):
        self.message.setText(text)

    def set_gesture(self, name: str | None, confidence: float = 0.0):
        """Текущий статический жест от ML (независимо от лога действий)."""
        if not name:
            self.gesture_value.setText("—")
            return
        try:
            pct = f"{float(confidence):.0%}"
        except (TypeError, ValueError):
            pct = "?"
        self.gesture_value.setText(f"{name}  ({pct})")

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
            btn = QPushButton(f"📦 {item}")
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #34495e; color: white; font-size: 12px;
                    padding: 6px; border-radius: 4px; text-align: left;
                }
                QPushButton:hover { background-color: #2980b9; }
            """)
            btn.clicked.connect(lambda checked, name=item: self.inventory_item_clicked.emit(name))
            self.inventory_layout.addWidget(btn)


    def set_camera_frame(self, base64_data: str):
        """Показать JPEG кадр из base64 (oil_gestures.ml.camera_frame)."""
        from PySide6.QtGui import QPixmap
        import base64
        import binascii

        try:
            data = base64.b64decode(base64_data)
        except (binascii.Error, ValueError):
            return

        pixmap = QPixmap()
        if not pixmap.loadFromData(data, "JPEG"):
            return

        self.camera_label.setPixmap(pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))