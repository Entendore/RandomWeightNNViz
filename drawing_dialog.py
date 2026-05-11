"""
Interactive dialog for drawing custom 2D datasets.
"""

import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont


class DrawingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Draw Custom 2D Dataset")
        self.setFixedSize(500, 550)
        self.X, self.y, self.drawing_class = [], [], 0

        layout = QVBoxLayout(self)

        # Class selector
        hl = QHBoxLayout()
        self.btn_class0 = QRadioButton("Class 0 (Blue)")
        self.btn_class0.setChecked(True)
        self.btn_class1 = QRadioButton("Class 1 (Red)")
        grp = QButtonGroup(self)
        grp.addButton(self.btn_class0, 0)
        grp.addButton(self.btn_class1, 1)
        grp.idClicked.connect(lambda idx: setattr(self, 'drawing_class', idx))
        hl.addWidget(self.btn_class0)
        hl.addWidget(self.btn_class1)
        layout.addLayout(hl)

        # Canvas
        self.canvas = QWidget()
        self.canvas.setFixedSize(480, 420)
        self.canvas.setStyleSheet("background-color: #1a1a2e; border: 2px solid #444;")
        layout.addWidget(self.canvas, alignment=Qt.AlignCenter)

        # Buttons
        hl2 = QHBoxLayout()
        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self._clear)
        btn_ok = QPushButton("Accept Dataset")
        btn_ok.clicked.connect(self.accept)
        hl2.addWidget(btn_clear)
        hl2.addWidget(btn_ok)
        layout.addLayout(hl2)

        # Mouse events
        self.canvas.mousePressEvent = self._on_click
        self.canvas.mouseMoveEvent = self._on_move
        self._pressed = False

    def _on_click(self, event):
        self._pressed = True
        self._add_point(event.position())

    def _on_move(self, event):
        if self._pressed and event.buttons() & (Qt.LeftButton | Qt.RightButton):
            c = 0 if event.buttons() & Qt.LeftButton else 1
            self.drawing_class = c
            self._add_point(event.position())

    def _add_point(self, pos):
        w, h = self.canvas.width(), self.canvas.height()
        self.X.append([(pos.x() / w) * 6 - 3, (pos.y() / h) * 6 - 3])
        self.y.append([self.drawing_class])
        self.canvas.update()

    def _clear(self):
        self.X, self.y = [], []
        self.canvas.update()

    def get_data(self):
        if len(self.X) == 0:
            return None, None
        return np.array(self.X), np.array(self.y)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.canvas.geometry(), QColor(26, 26, 46))
        w, h = self.canvas.width(), self.canvas.height()

        # Grid
        p.setPen(QPen(QColor(50, 50, 70), 0.5))
        for i in range(1, 6):
            p.drawLine(QPointF(i * w / 6, 0), QPointF(i * w / 6, h))
            p.drawLine(QPointF(0, i * h / 6), QPointF(w, i * h / 6))

        # Points
        for xi, yi in zip(self.X, self.y):
            cx, cy = ((xi[0] + 3) / 6) * w, ((xi[1] + 3) / 6) * h
            c = QColor(50, 200, 255) if yi[0] == 0 else QColor(255, 80, 80)
            p.setPen(QPen(QColor(255, 255, 255, 80), 1))
            p.setBrush(QBrush(c))
            p.drawEllipse(QPointF(cx, cy), 5, 5)
        p.end()