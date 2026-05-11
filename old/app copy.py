#!/usr/bin/env python3
"""
Pure Random Weight Search Visualizer
Randomize weights → evaluate → repeat → pick the best.
No training at all — can random weights alone solve the problem?
"""

import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QSpinBox, QGroupBox, QSplitter,
    QDoubleSpinBox, QGridLayout, QProgressBar, QSizePolicy, QCheckBox
)
from PySide6.QtCore import Qt, QTimer, QRectF, QPointF
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPainterPath,
    QLinearGradient, QRadialGradient, QPalette, QImage
)


# ─── Neural Network ───────────────────────────────────────────────────────────

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = list(layer_sizes)
        self.weights = []
        self.biases = []
        self.activations = []

    def randomize(self, method="he"):
        self.weights, self.biases = [], []
        for i in range(len(self.layer_sizes) - 1):
            fi, fo = self.layer_sizes[i], self.layer_sizes[i + 1]
            if method == "xavier":
                W = np.random.randn(fi, fo) * np.sqrt(2.0 / (fi + fo))
            elif method == "he":
                W = np.random.randn(fi, fo) * np.sqrt(2.0 / fi)
            elif method == "lecun":
                W = np.random.randn(fi, fo) * np.sqrt(1.0 / fi)
            elif method == "uniform":
                lim = np.sqrt(6.0 / (fi + fo))
                W = np.random.uniform(-lim, lim, (fi, fo))
            elif method == "normal_01":
                W = np.random.randn(fi, fo) * 0.1
            elif method == "normal_1":
                W = np.random.randn(fi, fo) * 1.0
            elif method == "large":
                W = np.random.randn(fi, fo) * 5.0
            elif method == "sparse":
                W = np.random.randn(fi, fo) * 0.01
                W[np.random.rand(fi, fo) < 0.8] = 0.0
            else:
                W = np.random.randn(fi, fo) * 0.5
            self.weights.append(W)
            self.biases.append(np.random.randn(1, fo) * 0.01)
        self.activations = []

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, X):
        self.activations = [X.copy()]
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            a = self._sigmoid(z) if i == len(self.weights) - 1 else np.maximum(0, z)
            self.activations.append(a.copy())
        return a

    def evaluate(self, X, y):
        out = self.forward(X)
        acc = float(np.mean((out > 0.5).astype(float) == y))
        eps = 1e-8
        loss = float(-np.mean(y * np.log(out + eps) + (1 - y) * np.log(1 - out + eps)))
        return acc, loss

    def copy_from(self, other):
        self.layer_sizes = list(other.layer_sizes)
        self.weights = [w.copy() for w in other.weights]
        self.biases = [b.copy() for b in other.biases]
        self.activations = [a.copy() for a in other.activations] if other.activations else []

    def weight_stats(self):
        all_w = np.concatenate([w.ravel() for w in self.weights])
        return {"mean": np.mean(all_w), "std": np.std(all_w),
                "min": np.min(all_w), "max": np.max(all_w)}


# ─── Data ─────────────────────────────────────────────────────────────────────

class DataGenerator:
    @staticmethod
    def generate(name, n=400):
        if name == "xor":
            X = np.random.randn(n, 2) * 0.5
            y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float).reshape(-1, 1)
        elif name == "circle":
            X = np.random.randn(n, 2) * 1.5
            y = (np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2) < 1.0).astype(float).reshape(-1, 1)
        elif name == "spiral":
            n2 = n // 2
            X, y = np.zeros((n, 2)), np.zeros((n, 1))
            for c in range(2):
                for i in range(n2):
                    r = i / n2 * 3
                    t = i / n2 * 4 * np.pi + c * np.pi
                    X[c * n2 + i] = [r * np.cos(t) + np.random.randn() * 0.15,
                                     r * np.sin(t) + np.random.randn() * 0.15]
                    y[c * n2 + i] = c
        elif name == "moons":
            n2 = n // 2
            th = np.linspace(0, np.pi, n2)
            X0 = np.column_stack([np.cos(th) + np.random.randn(n2) * 0.1,
                                   np.sin(th) + np.random.randn(n2) * 0.1])
            X1 = np.column_stack([1 - np.cos(th) + np.random.randn(n2) * 0.1,
                                   -np.sin(th) + 0.5 + np.random.randn(n2) * 0.1])
            X, y = np.vstack([X0, X1]), np.vstack([np.zeros((n2, 1)), np.ones((n2, 1))])
        else:
            n2 = n // 2
            X = np.vstack([np.random.randn(n2, 2) * 0.5 + [-1, -1],
                           np.random.randn(n2, 2) * 0.5 + [1, 1]])
            y = np.vstack([np.zeros((n2, 1)), np.ones((n2, 1))])
        return X, y


# ─── Network Viz ──────────────────────────────────────────────────────────────

class NetworkVizWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.nn = None
        self.label = "Current Candidate"
        self.is_best = False
        self.setMinimumSize(300, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_network(self, nn, label="Current Candidate", is_best=False):
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
            p.end(); return

        layers = self.nn.layer_sizes
        n_lay = len(layers)
        mx, my = 50, 40

        xs = [mx + i * (w - 2 * mx) / max(n_lay - 1, 1) for i in range(n_lay)]
        pos = []
        for li, sz in enumerate(layers):
            sp = min(28, (h - 2 * my - 20) / max(sz, 1))
            total = (sz - 1) * sp
            y0 = (h - total) / 2
            pos.append([(xs[li], y0 + j * sp) for j in range(sz)])

        # connections
        if self.nn.weights:
            wmax = max(np.max(np.abs(w)) for w in self.nn.weights) + 0.001
            for li in range(len(self.nn.weights)):
                W = self.nn.weights[li]
                for i in range(W.shape[0]):
                    for j in range(W.shape[1]):
                        val = W[i, j]
                        nv = val / wmax
                        alpha = int(min(abs(nv) * 160 + 12, 180))
                        color = QColor(80, 170, 255, alpha) if val > 0 else QColor(255, 85, 75, alpha)
                        pen = QPen(color, max(abs(nv) * 2.5, 0.25))
                        pen.setCapStyle(Qt.RoundCap)
                        p.setPen(pen)
                        x1, y1 = pos[li][i]; x2, y2 = pos[li + 1][j]
                        mid = (x1 + x2) / 2
                        path = QPainterPath(); path.moveTo(x1, y1)
                        path.cubicTo(mid, y1, mid, y2, x2, y2)
                        p.drawPath(path)

        # nodes
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
                    r, b = int(act_val * 220 + 30), int((1 - act_val) * 220 + 30)
                    nc = QColor(r, 55, b)
                else:
                    v = int(act_val * 170 + 60)
                    nc = QColor(v, v, min(v + 50, 255))

                grad = QRadialGradient(QPointF(nx, ny), 14)
                grad.setColorAt(0, QColor(nc.red(), nc.green(), nc.blue(), 70))
                grad.setColorAt(1, QColor(nc.red(), nc.green(), nc.blue(), 0))
                p.setPen(Qt.NoPen); p.setBrush(QBrush(grad))
                p.drawEllipse(QPointF(nx, ny), 14, 14)

                p.setPen(QPen(QColor(160, 160, 180, 140), 1))
                p.setBrush(QBrush(nc))
                p.drawEllipse(QPointF(nx, ny), 7, 7)

        # label
        if self.is_best:
            p.setPen(QColor(255, 215, 0))
            p.setFont(QFont("Segoe UI", 10, QFont.Bold))
        else:
            p.setPen(QColor(160, 160, 180))
            p.setFont(QFont("Segoe UI", 9))
        p.drawText(QRectF(0, 4, w, 18), Qt.AlignCenter, self.label)

        # layer labels
        p.setPen(QColor(110, 110, 130))
        p.setFont(QFont("Consolas", 7))
        names = ["In"] + [f"H{i+1}" for i in range(n_lay - 2)] + ["Out"]
        for li in range(n_lay):
            p.drawText(QRectF(xs[li] - 24, h - 18, 48, 14),
                       Qt.AlignCenter, f"{names[li]}({layers[li]})")

        # stats
        if self.nn.weights:
            s = self.nn.weight_stats()
            p.setPen(QColor(100, 100, 120)); p.setFont(QFont("Consolas", 7))
            p.drawText(6, h - 4, f"W μ={s['mean']:.2f} σ={s['std']:.2f}")
        p.end()


# ─── Decision Boundary ────────────────────────────────────────────────────────

class BoundaryWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.nn = None
        self.X = self.y = None
        self._img = None
        self._xr = self._yr = (-1, 1)
        self.label = "Decision Boundary"
        self.acc = None
        self.resolution = 80
        self.setMinimumSize(260, 260)

    def set_data(self, X, y):
        self.X, self.y = X, y

    def update_network(self, nn, label="Decision Boundary", acc=None):
        self.nn = nn; self.label = label; self.acc = acc
        self._rebuild()

    def _rebuild(self):
        if self.nn is None or self.X is None:
            self._img = None; self.update(); return
        xmn, xmx = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        ymn, ymx = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
        res = self.resolution
        xx, yy = np.meshgrid(np.linspace(xmn, xmx, res), np.linspace(ymn, ymx, res))
        Z = self.nn.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(res, res)
        self._xr, self._yr = (xmn, xmx), (ymn, ymx)

        img = QImage(res, res, QImage.Format_RGB32)
        for i in range(res):
            for j in range(res):
                v = Z[i, j]
                r = int(np.clip(v * 200 + 25, 0, 255))
                g = int(np.clip(min(v, 1 - v) * 4 * 50, 0, 255))
                b = int(np.clip((1 - v) * 200 + 25, 0, 255))
                img.setPixelColor(j, i, QColor(r, g, b))
        self._img = img
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        mg = 30
        p.fillRect(self.rect(), QColor(24, 24, 34))

        p.setPen(QColor(190, 190, 210))
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        txt = self.label
        if self.acc is not None:
            txt += f"  —  Acc: {self.acc * 100:.1f}%"
        p.drawText(QRectF(0, 3, w, 18), Qt.AlignCenter, txt)

        pw, ph = w - 2 * mg, h - 2 * mg - 10
        py0 = mg + 8

        if self._img:
            p.drawImage(QRectF(mg, py0, pw, ph), self._img)

        if self.X is not None and self.y is not None:
            xmn, xmx = self._xr; ymn, ymx = self._yr
            dx, dy = xmx - xmn, ymx - ymn
            for i in range(len(self.X)):
                px = mg + (self.X[i, 0] - xmn) / dx * pw
                py = py0 + (self.X[i, 1] - ymn) / dy * ph
                c = QColor(255, 210, 50) if self.y[i, 0] > 0.5 else QColor(50, 200, 255)
                p.setPen(QPen(QColor(0, 0, 0, 100), 0.6))
                p.setBrush(QBrush(c))
                p.drawEllipse(QPointF(px, py), 3, 3)

        p.setPen(QPen(QColor(70, 70, 90), 1)); p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(mg, py0, pw, ph))
        p.end()


# ─── Scatter History ──────────────────────────────────────────────────────────

METHOD_COLORS = {
    "he": QColor(80, 200, 120), "xavier": QColor(90, 160, 255),
    "lecun": QColor(0, 200, 200), "uniform": QColor(255, 210, 50),
    "normal_01": QColor(255, 160, 50), "normal_1": QColor(255, 90, 90),
    "large": QColor(220, 80, 220), "sparse": QColor(170, 130, 255),
}


class HistoryWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.attempts = []       # (index, method, acc, loss)
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
        self.attempts = []; self.best_idx = -1; self.best_acc = -1; self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        ml, mr, mt, mb = 52, 16, 30, 28
        p.fillRect(self.rect(), QColor(24, 24, 34))

        p.setPen(QColor(190, 190, 210))
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(QRectF(0, 3, w, 18), Qt.AlignCenter,
                   f"Accuracy per Attempt  ({len(self.attempts)} total)")

        px, py_, pw, ph = ml, mt, w - ml - mr, h - mt - mb

        # axes
        p.setPen(QPen(QColor(80, 80, 100), 1))
        p.drawLine(QPointF(px, py_ + ph), QPointF(px + pw, py_ + ph))
        p.drawLine(QPointF(px, py_), QPointF(px, py_ + ph))

        # y grid + labels
        p.setFont(QFont("Consolas", 7))
        for v in np.arange(0, 1.01, 0.25):
            y = py_ + ph - v * ph
            p.setPen(QPen(QColor(45, 45, 60), 0.5))
            p.drawLine(QPointF(px, y), QPointF(px + pw, y))
            p.setPen(QColor(120, 120, 140))
            p.drawText(QRectF(0, y - 6, ml - 4, 12), Qt.AlignRight | Qt.AlignVCenter,
                       f"{v:.0%}")

        # 50% reference
        p.setPen(QPen(QColor(80, 80, 50), 1, Qt.DashLine))
        y50 = py_ + ph * 0.5
        p.drawLine(QPointF(px, y50), QPointF(px + pw, y50))
        p.setPen(QColor(100, 100, 80)); p.setFont(QFont("Consolas", 6))
        p.drawText(QPointF(px + pw - 48, y50 - 3), "50% random")

        if not self.attempts:
            p.setPen(QColor(80, 80, 100)); p.setFont(QFont("Segoe UI", 9))
            p.drawText(self.rect(), Qt.AlignCenter, "Press Start to begin random search")
            p.end(); return

        n = len(self.attempts)
        # x labels
        p.setPen(QColor(120, 120, 140)); p.setFont(QFont("Consolas", 7))
        p.drawText(QRectF(px, py_ + ph + 3, pw, 12), Qt.AlignCenter, "Attempt #")

        # draw dots
        for ai, (idx, method, acc, loss) in enumerate(self.attempts):
            x = px + (idx / max(n - 1, 1)) * pw if n > 1 else px + pw / 2
            y = py_ + ph - acc * ph
            is_best = (ai == self.best_idx)
            c = METHOD_COLORS.get(method, QColor(180, 180, 200))
            if is_best:
                # glow
                grad = QRadialGradient(QPointF(x, y), 12)
                grad.setColorAt(0, QColor(255, 215, 0, 100))
                grad.setColorAt(1, QColor(255, 215, 0, 0))
                p.setPen(Qt.NoPen); p.setBrush(QBrush(grad))
                p.drawEllipse(QPointF(x, y), 12, 12)
                p.setPen(QPen(QColor(255, 255, 255), 1.5))
                p.setBrush(QBrush(QColor(255, 215, 0)))
                p.drawEllipse(QPointF(x, y), 5, 5)
            else:
                p.setPen(QPen(QColor(0, 0, 0, 60), 0.5))
                p.setBrush(QBrush(c))
                sz = 3.0
                p.drawEllipse(QPointF(x, y), sz, sz)

        # best label
        if self.best_idx >= 0:
            _, bm, ba, _ = self.attempts[self.best_idx]
            p.setPen(QColor(255, 215, 0)); p.setFont(QFont("Segoe UI", 8, QFont.Bold))
            p.drawText(QRectF(px, py_ + 2, pw, 14), Qt.AlignRight,
                       f"Best: {ba * 100:.1f}% (#{self.attempts[self.best_idx][0] + 1}, {bm})")

        # legend
        used = sorted(set(a[1] for a in self.attempts))
        ly = py_ + 15
        for m in used:
            c = METHOD_COLORS.get(m, QColor(180, 180, 200))
            p.setPen(Qt.NoPen); p.setBrush(QBrush(c))
            p.drawEllipse(QPointF(px + pw - 82, ly), 3.5, 3.5)
            p.setPen(QColor(150, 150, 170)); p.setFont(QFont("Consolas", 7))
            p.drawText(QPointF(px + pw - 74, ly + 3), m)
            ly += 12

        # stats
        accs = [a[2] for a in self.attempts]
        p.setPen(QColor(140, 140, 160)); p.setFont(QFont("Consolas", 7))
        p.drawText(QRectF(px, py_ + ph + 14, pw, 12), Qt.AlignCenter,
                   f"μ={np.mean(accs):.3f}  σ={np.std(accs):.3f}  "
                   f"median={np.median(accs):.3f}")
        p.end()


# ─── Histogram ────────────────────────────────────────────────────────────────

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
        self.accs = []; self.best_acc = -1; self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        ml, mr, mt, mb = 52, 16, 28, 22
        p.fillRect(self.rect(), QColor(24, 24, 34))

        p.setPen(QColor(190, 190, 210))
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(QRectF(0, 3, w, 18), Qt.AlignCenter, "Accuracy Distribution")

        px, py_, pw, ph = ml, mt, w - ml - mr, h - mt - mb

        if not self.accs:
            p.end(); return

        n_bins = min(20, max(5, len(self.accs) // 3))
        lo, hi = 0.0, 1.0
        bins = np.linspace(lo, hi, n_bins + 1)
        counts, _ = np.histogram(self.accs, bins=bins)
        max_c = max(counts.max(), 1)
        bw = pw / n_bins

        for i in range(n_bins):
            x = px + i * bw
            bh = counts[i] / max_c * ph
            mid_acc = (bins[i] + bins[i + 1]) / 2
            # color: red for bad, green for good
            r = int((1 - mid_acc) * 200 + 40)
            g = int(mid_acc * 200 + 40)
            b = 60
            c = QColor(r, g, b, 180)
            p.setPen(QPen(QColor(200, 200, 220, 60), 0.5))
            p.setBrush(QBrush(c))
            p.drawRect(QRectF(x + 1, py_ + ph - bh, bw - 2, bh))

        # best line
        if self.best_acc >= 0:
            bx = px + self.best_acc * pw
            p.setPen(QPen(QColor(255, 215, 0), 2, Qt.DashLine))
            p.drawLine(QPointF(bx, py_), QPointF(bx, py_ + ph))
            p.setPen(QColor(255, 215, 0)); p.setFont(QFont("Consolas", 7))
            p.drawText(QPointF(bx + 3, py_ + 10), f"Best {self.best_acc:.1%}")

        # 50% line
        x50 = px + 0.5 * pw
        p.setPen(QPen(QColor(100, 100, 70), 1, Qt.DotLine))
        p.drawLine(QPointF(x50, py_), QPointF(x50, py_ + ph))

        # y labels
        p.setPen(QColor(120, 120, 140)); p.setFont(QFont("Consolas", 7))
        p.drawText(QRectF(0, py_ - 2, ml - 4, 12), Qt.AlignRight | Qt.AlignVCenter,
                   str(max_c))
        p.drawText(QRectF(0, py_ + ph - 6, ml - 4, 12), Qt.AlignRight | Qt.AlignVCenter, "0")
        p.drawText(QRectF(px, py_ + ph + 4, pw, 12), Qt.AlignCenter, "Accuracy →")
        p.end()


# ─── Main Window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    METHODS = ["he", "xavier", "lecun", "uniform",
               "normal_01", "normal_1", "large", "sparse"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pure Random Weight Search — Can Random Weights Solve It?")
        self.setMinimumSize(1100, 720)
        self.resize(1400, 900)

        self.nn = None
        self.best_nn = None
        self.X = self.y = None
        self.running = False
        self.current_iter = 0
        self.total_iter = 100
        self.best_acc = -1
        self.current_acc = 0
        self.current_loss = 0

        self._build_ui()
        self._build_timers()
        self._full_reset()

    def _build_ui(self):
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(34, 34, 46))
        pal.setColor(QPalette.WindowText, QColor(200, 200, 220))
        pal.setColor(QPalette.Base, QColor(30, 30, 42))
        pal.setColor(QPalette.Text, QColor(200, 200, 220))
        pal.setColor(QPalette.Button, QColor(48, 48, 62))
        pal.setColor(QPalette.ButtonText, QColor(200, 200, 220))
        pal.setColor(QPalette.Highlight, QColor(80, 120, 200))
        self.setPalette(pal)

        self.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #555; border-radius: 5px;
                        margin-top: 14px; padding-top: 16px; color: #c8c8dc; }
            QGroupBox::title { subcontrol-origin: margin; padding: 0 6px; }
            QComboBox, QSpinBox, QDoubleSpinBox { padding: 3px; border: 1px solid #555;
                        border-radius: 3px; background: #38384c; color: #ddd; }
            QProgressBar { border: 1px solid #555; border-radius: 3px; text-align: center;
                           background: #2a2a3a; color: #ddd; font-weight: bold; }
            QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                              stop:0 #6a50a0, stop:1 #4090c0); border-radius: 2px; }
            QCheckBox { color: #ccc; } QLabel { color: #bbb; }
        """)

        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(6, 6, 6, 6); root.setSpacing(6)

        # LEFT
        left = QWidget(); left.setFixedWidth(250)
        lv = QVBoxLayout(left); lv.setContentsMargins(0, 0, 0, 0); lv.setSpacing(4)

        # dataset
        g = QGroupBox("Dataset"); gl = QVBoxLayout(g)
        gl.addWidget(QLabel("Type:"))
        self.dataset_cb = QComboBox()
        self.dataset_cb.addItems(["xor", "circle", "spiral", "moons", "gaussian"])
        self.dataset_cb.currentTextChanged.connect(self._on_data_changed)
        gl.addWidget(self.dataset_cb)
        hl = QHBoxLayout(); hl.addWidget(QLabel("Samples:"))
        self.nspin = QSpinBox(); self.nspin.setRange(50, 2000); self.nspin.setValue(400)
        self.nspin.setSingleStep(50); self.nspin.valueChanged.connect(self._on_data_changed)
        hl.addWidget(self.nspin); gl.addLayout(hl); lv.addWidget(g)

        # arch
        g = QGroupBox("Architecture"); gl = QGridLayout(g)
        gl.addWidget(QLabel("Input: 2 (fixed)"), 0, 0, 1, 2)
        for r, (label, default) in enumerate([("Hidden 1:", 8), ("Hidden 2:", 8), ("Hidden 3:", 0)], 1):
            gl.addWidget(QLabel(label), r, 0)
            sp = QSpinBox(); sp.setRange(0, 32); sp.setValue(default)
            sp.valueChanged.connect(self._on_arch_changed)
            gl.addWidget(sp, r, 1); setattr(self, f"h{r}_spin", sp)
        gl.addWidget(QLabel("Output: 1 (fixed)"), 4, 0, 1, 2)
        lv.addWidget(g)

        # randomization
        g = QGroupBox("Randomization"); gl = QVBoxLayout(g)
        gl.addWidget(QLabel("Init method:"))
        self.method_cb = QComboBox(); self.method_cb.addItems(self.METHODS)
        gl.addWidget(self.method_cb)
        self.compare_cb = QCheckBox("Cycle through ALL methods")
        self.compare_cb.setToolTip("Each attempt uses the next init method in rotation")
        gl.addWidget(self.compare_cb)
        lv.addWidget(g)

        # search
        g = QGroupBox("Random Search"); gl = QVBoxLayout(g)
        hl = QHBoxLayout(); hl.addWidget(QLabel("Iterations:"))
        self.iter_spin = QSpinBox(); self.iter_spin.setRange(10, 10000)
        self.iter_spin.setValue(200); self.iter_spin.setSingleStep(50)
        hl.addWidget(self.iter_spin); gl.addLayout(hl)
        hl = QHBoxLayout(); hl.addWidget(QLabel("Speed:"))
        self.speed_spin = QSpinBox(); self.speed_spin.setRange(1, 200)
        self.speed_spin.setValue(3); self.speed_spin.setSuffix(" / tick")
        hl.addWidget(self.speed_spin); gl.addLayout(hl)

        self.start_btn = QPushButton("🎲  Start Random Search")
        self.start_btn.setStyleSheet(
            "QPushButton{background:#2a7090;color:#fff;padding:8px;"
            "border-radius:4px;font-weight:bold;font-size:12px}"
            "QPushButton:hover{background:#3a80a0}")
        self.start_btn.clicked.connect(self._toggle_run)
        gl.addWidget(self.start_btn)

        self.step_btn = QPushButton("→  Single Step")
        self.step_btn.setStyleSheet(
            "QPushButton{background:#505068;color:#ddd;padding:5px;"
            "border-radius:3px}"
            "QPushButton:hover{background:#606080}")
        self.step_btn.clicked.connect(self._single_step)
        gl.addWidget(self.step_btn)

        self.reset_btn = QPushButton("↺  Full Reset")
        self.reset_btn.setStyleSheet(
            "QPushButton{background:#804040;color:#fff;padding:5px;"
            "border-radius:3px}"
            "QPushButton:hover{background:#905050}")
        self.reset_btn.clicked.connect(self._full_reset)
        gl.addWidget(self.reset_btn)

        self.progress = QProgressBar(); self.progress.setVisible(False)
        gl.addWidget(self.progress)
        lv.addWidget(g)

        # stats
        g = QGroupBox("Live Stats"); gl = QVBoxLayout(g)
        self.cur_label = QLabel("Current:  —")
        self.cur_label.setStyleSheet("color:#aaa;font-size:11px")
        gl.addWidget(self.cur_label)
        self.best_label = QLabel("Best:  —")
        self.best_label.setStyleSheet("color:#ffd700;font-size:12px;font-weight:bold")
        gl.addWidget(self.best_label)
        self.iter_label = QLabel("Iteration: 0")
        self.iter_label.setStyleSheet("color:#999;font-size:10px")
        gl.addWidget(self.iter_label)
        self.pct_label = QLabel("")
        self.pct_label.setStyleSheet("color:#888;font-size:9px")
        gl.addWidget(self.pct_label)
        lv.addWidget(g)
        lv.addStretch()
        root.addWidget(left)

        # RIGHT
        splitter = QSplitter(Qt.Vertical)

        # row 1: two boundaries side by side
        row1 = QSplitter(Qt.Horizontal)
        self.boundary_cur = BoundaryWidget()
        self.boundary_best = BoundaryWidget()
        row1.addWidget(self.boundary_cur)
        row1.addWidget(self.boundary_best)
        row1.setSizes([500, 500])

        # row 2: scatter + histogram
        row2 = QSplitter(Qt.Horizontal)
        self.history = HistoryWidget()
        self.histogram = HistogramWidget()
        row2.addWidget(self.history)
        row2.addWidget(self.histogram)
        row2.setSizes([500, 300])

        # row 3: network viz
        self.net_viz = NetworkVizWidget()

        splitter.addWidget(row1)
        splitter.addWidget(row2)
        splitter.addWidget(self.net_viz)
        splitter.setSizes([320, 220, 280])

        root.addWidget(splitter, 1)
        self.statusBar().setStyleSheet("color:#aaa;")
        self.statusBar().showMessage("Ready — press Start to search for good random weights!")

    def _build_timers(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(25)

    # ── helpers ──

    def _layer_sizes(self):
        layers = [2]
        for sp in (self.h1_spin, self.h2_spin, self.h3_spin):
            v = sp.value()
            if v > 0: layers.append(v)
        layers.append(1)
        return layers

    def _get_method(self):
        if self.compare_cb.isChecked():
            return self.METHODS[self.current_iter % len(self.METHODS)]
        return self.method_cb.currentText()

    def _full_reset(self):
        self._stop()
        self.current_iter = 0; self.best_acc = -1; self.best_nn = None
        self.nn = NeuralNetwork(self._layer_sizes())
        self.nn.randomize(self.method_cb.currentText())
        self.X, self.y = DataGenerator.generate(self.dataset_cb.currentText(), self.nspin.value())
        acc, loss = self.nn.evaluate(self.X, self.y)
        self.current_acc = acc; self.current_loss = loss

        self.boundary_cur.set_data(self.X, self.y)
        self.boundary_best.set_data(self.X, self.y)
        self.boundary_cur.update_network(self.nn, "Current Candidate", acc)
        self.boundary_best.update_network(None, "Best Found (none yet)")
        self.net_viz.set_network(self.nn, "Current Candidate")
        self.history.clear(); self.histogram.clear()
        self.cur_label.setText(f"Current:  Acc {acc:.1%}  Loss {loss:.4f}")
        self.best_label.setText("Best:  —")
        self.iter_label.setText("Iteration: 0")
        self.pct_label.setText("")
        self.statusBar().showMessage("Reset — press Start to begin random search")

    def _refresh_current(self, method):
        acc, loss = self.nn.evaluate(self.X, self.y)
        self.current_acc = acc; self.current_loss = loss
        is_new_best = acc > self.best_acc

        if is_new_best:
            self.best_acc = acc
            if self.best_nn is None:
                self.best_nn = NeuralNetwork(self._layer_sizes())
            self.best_nn.copy_from(self.nn)
            self.boundary_best.update_network(self.best_nn, f"★ Best Found", acc)

        self.boundary_cur.update_network(self.nn, f"#{self.current_iter + 1} ({method})", acc)
        self.net_viz.set_network(self.nn, f"Candidate #{self.current_iter + 1}  [{method}]",
                                 is_best=is_new_best)

        self.history.add(self.current_iter, method, acc, loss)
        self.histogram.add(acc)

        self.cur_label.setText(f"Current:  Acc {acc:.1%}  Loss {loss:.4f}")
        if self.best_nn:
            self.best_label.setText(f"Best:  Acc {self.best_acc:.1%}  (iter #{self.history.attempts[self.history.best_idx][0] + 1})")
        self.iter_label.setText(f"Iteration: {self.current_iter + 1} / {self.total_iter}")

        accs = [a[2] for a in self.history.attempts]
        above_70 = sum(1 for a in accs if a > 0.7)
        self.pct_label.setText(
            f"Mean: {np.mean(accs):.1%}  |  >70%: {above_70}/{len(accs)}  |  "
            f"Median: {np.median(accs):.1%}")

        if is_new_best:
            self.statusBar().showMessage(
                f"🎉 NEW BEST at #{self.current_iter + 1}!  Acc = {acc:.1%}  ({method})")
        else:
            self.statusBar().showMessage(
                f"#{self.current_iter + 1}  Acc={acc:.1%}  Best={self.best_acc:.1%}  ({method})")

    # ── slots ──

    def _on_data_changed(self, *_):
        self._full_reset()

    def _on_arch_changed(self, *_):
        self._full_reset()

    def _toggle_run(self):
        if self.running:
            self._stop()
        else:
            self._start()

    def _start(self):
        self.running = True
        self.total_iter = self.iter_spin.value()
        self.progress.setVisible(True)
        self.progress.setRange(0, self.total_iter)
        self.progress.setValue(self.current_iter)
        self.start_btn.setText("⏹  Stop")
        self.start_btn.setStyleSheet(
            "QPushButton{background:#c09020;color:#fff;padding:8px;"
            "border-radius:4px;font-weight:bold;font-size:12px}"
            "QPushButton:hover{background:#d0a030}")
        self.timer.start()

    def _stop(self):
        self.timer.stop(); self.running = False
        self.start_btn.setText("🎲  Start Random Search")
        self.start_btn.setStyleSheet(
            "QPushButton{background:#2a7090;color:#fff;padding:8px;"
            "border-radius:4px;font-weight:bold;font-size:12px}"
            "QPushButton:hover{background:#3a80a0}")
        if self.current_iter >= self.total_iter:
            self.progress.setVisible(False)
            self.statusBar().showMessage(
                f"Search complete!  Best accuracy: {self.best_acc:.1%}  "
                f"after {self.current_iter} random attempts")

    def _tick(self):
        if not self.running:
            return
        speed = self.speed_spin.value()
        for _ in range(speed):
            if self.current_iter >= self.total_iter:
                self._stop(); return
            method = self._get_method()
            self.nn = NeuralNetwork(self._layer_sizes())
            self.nn.randomize(method)
            self._refresh_current(method)
            self.current_iter += 1
            self.progress.setValue(self.current_iter)

    def _single_step(self):
        if self.running:
            return
        self.total_iter = self.iter_spin.value()
        method = self._get_method()
        self.nn = NeuralNetwork(self._layer_sizes())
        self.nn.randomize(method)
        self._refresh_current(method)
        self.current_iter += 1


# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())