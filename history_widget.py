"""
Scatter-history chart showing accuracy per attempt with method-based coloring.
"""

import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QRadialGradient
)

METHOD_COLORS = {
    "he": QColor(80, 200, 120),
    "xavier": QColor(90, 160, 255),
    "lecun": QColor(0, 200, 200),
    "uniform": QColor(255, 210, 50),
    "normal_01": QColor(255, 160, 50),
    "normal_1": QColor(255, 90, 90),
    "large": QColor(220, 80, 220),
    "sparse": QColor(170, 130, 255),
}


class HistoryWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.attempts = []
        self.best_idx = -1
        self.best_acc = -1
        self.setMinimumSize(280, 180)

    def add(self, idx, method, acc, loss):
        self.attempts.append((idx, method, acc, loss))
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_idx = len(self.attempts) - 1
        self.update()

    def clear(self):
        self.attempts = []
        self.best_idx = -1
        self.best_acc = -1
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        ml, mr, mt, mb = 52, 16, 30, 28
        p.fillRect(self.rect(), QColor(24, 24, 34))

        # Title
        p.setPen(QColor(190, 190, 210))
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(QRectF(0, 3, w, 18), Qt.AlignCenter,
                   f"Accuracy per Attempt ({len(self.attempts)})")

        # Plot area
        px, py_, pw, ph = ml, mt, w - ml - mr, h - mt - mb
        p.setPen(QPen(QColor(80, 80, 100), 1))
        p.drawLine(QPointF(px, py_ + ph), QPointF(px + pw, py_ + ph))
        p.drawLine(QPointF(px, py_), QPointF(px, py_ + ph))

        # Y-axis labels & grid
        p.setFont(QFont("Consolas", 7))
        for v in np.arange(0, 1.01, 0.25):
            y = py_ + ph - v * ph
            p.setPen(QPen(QColor(45, 45, 60), 0.5))
            p.drawLine(QPointF(px, y), QPointF(px + pw, y))
            p.setPen(QColor(120, 120, 140))
            p.drawText(QRectF(0, y - 6, ml - 4, 12),
                       Qt.AlignRight | Qt.AlignVCenter, f"{v:.0%}")

        # 50% dashed line
        p.setPen(QPen(QColor(80, 80, 50), 1, Qt.DashLine))
        y50 = py_ + ph * 0.5
        p.drawLine(QPointF(px, y50), QPointF(px + pw, y50))

        if not self.attempts:
            p.end()
            return

        # Data points
        n = len(self.attempts)
        for ai, (idx, method, acc, loss) in enumerate(self.attempts):
            x = px + (idx / max(n - 1, 1)) * pw if n > 1 else px + pw / 2
            y = py_ + ph - acc * ph
            is_best = (ai == self.best_idx)
            c = METHOD_COLORS.get(method, QColor(180, 180, 200))

            if is_best:
                grad = QRadialGradient(QPointF(x, y), 12)
                grad.setColorAt(0, QColor(255, 215, 0, 100))
                grad.setColorAt(1, QColor(255, 215, 0, 0))
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(grad))
                p.drawEllipse(QPointF(x, y), 12, 12)
                p.setPen(QPen(QColor(255, 255, 255), 1.5))
                p.setBrush(QBrush(QColor(255, 215, 0)))
                p.drawEllipse(QPointF(x, y), 5, 5)
            else:
                p.setPen(QPen(QColor(0, 0, 0, 60), 0.5))
                p.setBrush(QBrush(c))
                p.drawEllipse(QPointF(x, y), 3, 3)

        # Best label
        if self.best_idx >= 0:
            p.setPen(QColor(255, 215, 0))
            p.setFont(QFont("Segoe UI", 8, QFont.Bold))
            p.drawText(QRectF(px, py_ + 2, pw, 14),
                       Qt.AlignRight, f"Best: {self.best_acc * 100:.1f}%")
        p.end()