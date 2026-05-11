"""
Accuracy distribution histogram widget.
"""

import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont


class HistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.accs = []
        self.best_acc = -1
        self.setMinimumSize(280, 140)

    def add(self, acc):
        self.accs.append(acc)
        if acc > self.best_acc:
            self.best_acc = acc
        self.update()

    def clear(self):
        self.accs = []
        self.best_acc = -1
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        ml, mr, mt, mb = 52, 16, 28, 22
        p.fillRect(self.rect(), QColor(24, 24, 34))

        # Title
        p.setPen(QColor(190, 190, 210))
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(QRectF(0, 3, w, 18), Qt.AlignCenter, "Accuracy Distribution")

        px, py_, pw, ph = ml, mt, w - ml - mr, h - mt - mb

        if not self.accs:
            p.end()
            return

        # Bins
        n_bins = min(20, max(5, len(self.accs) // 3))
        bins = np.linspace(0, 1, n_bins + 1)
        counts, _ = np.histogram(self.accs, bins=bins)
        max_c = max(counts.max(), 1)
        bw = pw / n_bins

        for i in range(n_bins):
            x = px + i * bw
            bh = counts[i] / max_c * ph
            mid_acc = (bins[i] + bins[i + 1]) / 2
            c = QColor(int((1 - mid_acc) * 200 + 40), int(mid_acc * 200 + 40), 60, 180)
            p.setPen(QPen(QColor(200, 200, 220, 60), 0.5))
            p.setBrush(QBrush(c))
            p.drawRect(QRectF(x + 1, py_ + ph - bh, bw - 2, bh))

        # Best marker
        if self.best_acc >= 0:
            bx = px + self.best_acc * pw
            p.setPen(QPen(QColor(255, 215, 0), 2, Qt.DashLine))
            p.drawLine(QPointF(bx, py_), QPointF(bx, py_ + ph))
            p.setPen(QColor(255, 215, 0))
            p.setFont(QFont("Consolas", 7))
            p.drawText(QPointF(bx + 3, py_ + 10), f"Best {self.best_acc:.1%}")

        p.end()