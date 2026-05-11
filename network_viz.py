"""
Neural network architecture visualization widget.
Draws nodes, connections (curved), and activation-based coloring.
"""

import numpy as np
from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPainterPath
)


class NetworkVizWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.nn = None
        self.label = ""
        self.is_best = False
        self.setMinimumSize(300, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_network(self, nn, label="", is_best=False):
        self.nn = nn
        self.label = label
        self.is_best = is_best
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(24, 24, 34))

        if not self.nn:
            p.end()
            return

        layers = self.nn.layer_sizes
        n_lay = len(layers)
        mx, my = 50, 40
        xs = [mx + i * (w - 2 * mx) / max(n_lay - 1, 1) for i in range(n_lay)]

        # Compute node positions
        pos = []
        for li, sz in enumerate(layers):
            sp = min(24, (h - 2 * my - 20) / max(sz, 1))
            y0 = (h - (sz - 1) * sp) / 2
            pos.append([(xs[li], y0 + j * sp) for j in range(sz)])

        # Draw connections
        if self.nn.weights:
            wmax = max(np.max(np.abs(w)) for w in self.nn.weights) + 0.001
            for li in range(len(self.nn.weights)):
                W = self.nn.weights[li]
                for i in range(W.shape[0]):
                    for j in range(W.shape[1]):
                        val = W[i, j]
                        nv = val / wmax
                        alpha = int(min(abs(nv) * 160 + 12, 160))
                        color = QColor(80, 170, 255, alpha) if val > 0 else QColor(255, 85, 75, alpha)
                        pen = QPen(color, max(abs(nv) * 2.0, 0.2))
                        pen.setCapStyle(Qt.RoundCap)
                        p.setPen(pen)
                        x1, y1 = pos[li][i]
                        x2, y2 = pos[li + 1][j]
                        mid = (x1 + x2) / 2
                        path = QPainterPath()
                        path.moveTo(x1, y1)
                        path.cubicTo(mid, y1, mid, y2, x2, y2)
                        p.drawPath(path)

        # Draw nodes
        for li in range(n_lay):
            for ni, (nx, ny) in enumerate(pos[li]):
                act_val = 0.5
                if self.nn.activations and li < len(self.nn.activations):
                    a = self.nn.activations[li]
                    if ni < a.shape[1]:
                        act_val = float(np.clip(np.mean(a[:, ni]), 0, 1))

                if li == 0:
                    nc = QColor(100, 190, 255)
                elif li == n_lay - 1:
                    nc = QColor(int(act_val * 220 + 30), 55, int((1 - act_val) * 220 + 30))
                else:
                    v = int(act_val * 170 + 60)
                    nc = QColor(v, v, min(v + 50, 255))

                p.setPen(QPen(QColor(160, 160, 180, 140), 1))
                p.setBrush(QBrush(nc))
                radius = 6 if max(layers) > 16 else 7
                p.drawEllipse(QPointF(nx, ny), radius, radius)

        # Label
        if self.is_best:
            p.setPen(QColor(255, 215, 0))
            p.setFont(QFont("Segoe UI", 10, QFont.Bold))
        else:
            p.setPen(QColor(160, 160, 180))
            p.setFont(QFont("Segoe UI", 9))
        p.drawText(QRectF(0, 4, w, 18), Qt.AlignCenter, self.label)
        p.end()