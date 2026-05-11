#!/usr/bin/env python3
"""
Pure Random Weight Search Visualizer + Advanced Features + Auto Full-Run MP4 Exporter
Randomize weights → evaluate → repeat → pick the best.
Now includes Confusion Matrices, Top-K tracking, and Automated Video Generation!
"""

import sys
import os
import datetime
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QSpinBox, QGroupBox, QSplitter,
    QDoubleSpinBox, QGridLayout, QProgressBar, QSizePolicy, QCheckBox,
    QFileDialog, QDialog, QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt, QTimer, QRectF, QPointF
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPainterPath,
    QLinearGradient, QRadialGradient, QPalette, QImage, QPixmap
)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


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
            if method == "xavier": W = np.random.randn(fi, fo) * np.sqrt(2.0 / (fi + fo))
            elif method == "he": W = np.random.randn(fi, fo) * np.sqrt(2.0 / fi)
            elif method == "lecun": W = np.random.randn(fi, fo) * np.sqrt(1.0 / fi)
            elif method == "uniform":
                lim = np.sqrt(6.0 / (fi + fo)); W = np.random.uniform(-lim, lim, (fi, fo))
            elif method == "normal_01": W = np.random.randn(fi, fo) * 0.1
            elif method == "normal_1": W = np.random.randn(fi, fo) * 1.0
            elif method == "large": W = np.random.randn(fi, fo) * 5.0
            elif method == "sparse":
                W = np.random.randn(fi, fo) * 0.01; W[np.random.rand(fi, fo) < 0.8] = 0.0
            else: W = np.random.randn(fi, fo) * 0.5
            self.weights.append(W); self.biases.append(np.random.randn(1, fo) * 0.01)
        self.activations = []

    @staticmethod
    def _sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, X):
        self.activations = [X.copy()]; a = X
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


# ─── Data Generator ───────────────────────────────────────────────────────────

class DataGenerator:
    @staticmethod
    def get_shape(name):
        if name == "digits_4x4": return 16, 3
        if name == "hand_signs": return 5, 4
        return 2, 1

    @staticmethod
    def generate(name, n=400):
        if name == "xor":
            X = np.random.randn(n, 2) * 0.5
            y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float).reshape(-1, 1)
        elif name == "circle":
            X = np.random.randn(n, 2) * 1.5
            y = (np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2) < 1.0).astype(float).reshape(-1, 1)
        elif name == "spiral":
            n2 = n // 2; X, y = np.zeros((n, 2)), np.zeros((n, 1))
            for c in range(2):
                for i in range(n2):
                    r = i / n2 * 3; t = i / n2 * 4 * np.pi + c * np.pi
                    X[c * n2 + i] = [r * np.cos(t) + np.random.randn() * 0.15, r * np.sin(t) + np.random.randn() * 0.15]
                    y[c * n2 + i] = c
        elif name == "moons":
            n2 = n // 2; th = np.linspace(0, np.pi, n2)
            X0 = np.column_stack([np.cos(th) + np.random.randn(n2) * 0.1, np.sin(th) + np.random.randn(n2) * 0.1])
            X1 = np.column_stack([1 - np.cos(th) + np.random.randn(n2) * 0.1, -np.sin(th) + 0.5 + np.random.randn(n2) * 0.1])
            X, y = np.vstack([X0, X1]), np.vstack([np.zeros((n2, 1)), np.ones((n2, 1))])
        elif name == "digits_4x4":
            h_line = np.array([1,1,1,1, 0,0,0,0, 0,0,0,0, 0,0,0,0])
            v_line = np.array([1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0])
            d_line = np.array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1])
            classes = [h_line, v_line, d_line]; X, y = [], []
            for _ in range(n):
                c = np.random.randint(0, 3); X.append(np.clip(classes[c] + np.random.randn(16) * 0.3, 0, 1))
                lab = np.zeros(3); lab[c] = 1; y.append(lab)
            X, y = np.array(X), np.array(y)
        elif name == "hand_signs":
            fist = np.array([0,0,0,0,0]); peace = np.array([0,1,1,0,0])
            point = np.array([0,1,0,0,0]); open_h = np.array([1,1,1,1,1])
            classes = [fist, peace, point, open_h]; X, y = [], []
            for _ in range(n):
                c = np.random.randint(0, 4); X.append(np.clip(classes[c] + np.random.randn(5) * 0.2, 0, 1))
                lab = np.zeros(4); lab[c] = 1; y.append(lab)
            X, y = np.array(X), np.array(y)
        else:
            n2 = n // 2
            X = np.vstack([np.random.randn(n2, 2) * 0.5 + [-1, -1], np.random.randn(n2, 2) * 0.5 + [1, 1]])
            y = np.vstack([np.zeros((n2, 1)), np.ones((n2, 1))])
        return X, y


# ─── Interactive Drawing Dialog ────────────────────────────────────────────────

class DrawingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent); self.setWindowTitle("Draw Custom 2D Dataset"); self.setFixedSize(500, 550)
        self.X, self.y, self.drawing_class = [], [], 0
        layout = QVBoxLayout(self)
        hl = QHBoxLayout()
        self.btn_class0 = QRadioButton("Class 0 (Blue)"); self.btn_class0.setChecked(True)
        self.btn_class1 = QRadioButton("Class 1 (Red)")
        grp = QButtonGroup(self); grp.addButton(self.btn_class0, 0); grp.addButton(self.btn_class1, 1)
        grp.idClicked.connect(lambda idx: setattr(self, 'drawing_class', idx))
        hl.addWidget(self.btn_class0); hl.addWidget(self.btn_class1); layout.addLayout(hl)
        self.canvas = QWidget(); self.canvas.setFixedSize(480, 420)
        self.canvas.setStyleSheet("background-color: #1a1a2e; border: 2px solid #444;"); layout.addWidget(self.canvas, alignment=Qt.AlignCenter)
        hl2 = QHBoxLayout(); btn_clear = QPushButton("Clear All"); btn_clear.clicked.connect(self._clear)
        btn_ok = QPushButton("Accept Dataset"); btn_ok.clicked.connect(self.accept)
        hl2.addWidget(btn_clear); hl2.addWidget(btn_ok); layout.addLayout(hl2)
        self.canvas.mousePressEvent = self._on_click; self.canvas.mouseMoveEvent = self._on_move; self._pressed = False

    def _on_click(self, event): self._pressed = True; self._add_point(event.position())
    def _on_move(self, event):
        if self._pressed and event.buttons() & (Qt.LeftButton | Qt.RightButton):
            c = 0 if event.buttons() & Qt.LeftButton else 1; self.drawing_class = c; self._add_point(event.position())
    def _add_point(self, pos):
        w, h = self.canvas.width(), self.canvas.height()
        self.X.append([(pos.x() / w) * 6 - 3, (pos.y() / h) * 6 - 3]); self.y.append([self.drawing_class]); self.canvas.update()
    def _clear(self): self.X, self.y = [], []; self.canvas.update()
    def get_data(self):
        if len(self.X) == 0: return None, None
        return np.array(self.X), np.array(self.y)
    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.canvas.geometry(), QColor(26, 26, 46))
        w, h = self.canvas.width(), self.canvas.height()
        p.setPen(QPen(QColor(50, 50, 70), 0.5))
        for i in range(1, 6): p.drawLine(QPointF(i*w/6, 0), QPointF(i*w/6, h)); p.drawLine(QPointF(0, i*h/6), QPointF(w, i*h/6))
        for xi, yi in zip(self.X, self.y):
            cx, cy = ((xi[0]+3)/6)*w, ((xi[1]+3)/6)*h
            c = QColor(50, 200, 255) if yi[0] == 0 else QColor(255, 80, 80)
            p.setPen(QPen(QColor(255,255,255,80), 1)); p.setBrush(QBrush(c)); p.drawEllipse(QPointF(cx, cy), 5, 5)
        p.end()


# ─── Network Viz ──────────────────────────────────────────────────────────────

class NetworkVizWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.nn = None; self.label = ""; self.is_best = False
        self.setMinimumSize(300, 240); self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_network(self, nn, label="", is_best=False):
        self.nn = nn; self.label = label; self.is_best = is_best; self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height(); p.fillRect(self.rect(), QColor(24, 24, 34))
        if not self.nn: p.end(); return
        layers = self.nn.layer_sizes; n_lay = len(layers); mx, my = 50, 40
        xs = [mx + i * (w - 2 * mx) / max(n_lay - 1, 1) for i in range(n_lay)]
        pos = []
        for li, sz in enumerate(layers):
            sp = min(24, (h - 2 * my - 20) / max(sz, 1)); y0 = (h - (sz - 1) * sp) / 2
            pos.append([(xs[li], y0 + j * sp) for j in range(sz)])
        if self.nn.weights:
            wmax = max(np.max(np.abs(w)) for w in self.nn.weights) + 0.001
            for li in range(len(self.nn.weights)):
                W = self.nn.weights[li]
                for i in range(W.shape[0]):
                    for j in range(W.shape[1]):
                        val = W[i, j]; nv = val / wmax
                        alpha = int(min(abs(nv) * 160 + 12, 160))
                        color = QColor(80, 170, 255, alpha) if val > 0 else QColor(255, 85, 75, alpha)
                        pen = QPen(color, max(abs(nv) * 2.0, 0.2)); pen.setCapStyle(Qt.RoundCap); p.setPen(pen)
                        x1, y1 = pos[li][i]; x2, y2 = pos[li + 1][j]; mid = (x1 + x2) / 2
                        path = QPainterPath(); path.moveTo(x1, y1); path.cubicTo(mid, y1, mid, y2, x2, y2); p.drawPath(path)
        for li in range(n_lay):
            for ni, (nx, ny) in enumerate(pos[li]):
                act_val = 0.5
                if self.nn.activations and li < len(self.nn.activations):
                    a = self.nn.activations[li]
                    if ni < a.shape[1]: act_val = float(np.clip(np.mean(a[:, ni]), 0, 1))
                if li == 0: nc = QColor(100, 190, 255)
                elif li == n_lay - 1: nc = QColor(int(act_val * 220 + 30), 55, int((1 - act_val) * 220 + 30))
                else: v = int(act_val * 170 + 60); nc = QColor(v, v, min(v + 50, 255))
                p.setPen(QPen(QColor(160, 160, 180, 140), 1)); p.setBrush(QBrush(nc))
                p.drawEllipse(QPointF(nx, ny), 6 if max(layers) > 16 else 7, 6 if max(layers) > 16 else 7)
        if self.is_best: p.setPen(QColor(255, 215, 0)); p.setFont(QFont("Segoe UI", 10, QFont.Bold))
        else: p.setPen(QColor(160, 160, 180)); p.setFont(QFont("Segoe UI", 9))
        p.drawText(QRectF(0, 4, w, 18), Qt.AlignCenter, self.label); p.end()


# ─── Universal Data Viz ───────────────────────────────────────────────────────

HAND_LABELS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
DIGIT_LABELS = ["H-Line", "V-Line", "Diag"]
HAND_CLASSES = ["Fist", "Peace", "Point", "Open"]

class DataVizWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.nn = None; self.X = self.y = None; self._img = None
        self._xr = self._yr = (-1, 1); self.label = ""; self.acc = None; self.resolution = 60
        self.setMinimumSize(260, 260)

    def set_data(self, X, y): self.X, self.y = X, y
    def update_network(self, nn, label="", acc=None):
        self.nn = nn; self.label = label; self.acc = acc; self._rebuild()

    def _rebuild(self):
        if self.nn is None or self.X is None: self._img = None; self.update(); return
        if self.X.shape[1] == 2:
            xmn, xmx = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
            ymn, ymx = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
            res = self.resolution
            xx, yy = np.meshgrid(np.linspace(xmn, xmx, res), np.linspace(ymn, ymx, res))
            Z = self.nn.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(res, res, -1)
            self._xr, self._yr = (xmn, xmx), (ymn, ymx)
            img = QImage(res, res, QImage.Format_RGB32); n_out = Z.shape[2]
            colors = [QColor(50,200,255), QColor(255,80,80), QColor(80,255,80), QColor(255,200,50), QColor(200,80,255)]
            for i in range(res):
                for j in range(res):
                    if n_out == 1:
                        v = Z[i, j, 0]
                        img.setPixelColor(j, i, QColor(int(np.clip(v*200+25,0,255)), int(np.clip(min(v,1-v)*4*50,0,255)), int(np.clip((1-v)*200+25,0,255))))
                    else:
                        mx_idx = np.argmax(Z[i, j]); img.setPixelColor(j, i, colors[mx_idx % len(colors)])
            self._img = img
        else: self._img = None
        self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height(); p.fillRect(self.rect(), QColor(24, 24, 34))
        txt = self.label
        if self.acc is not None: txt += f"  Acc: {self.acc * 100:.1f}%"
        p.setPen(QColor(190, 190, 210)); p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(QRectF(0, 3, w, 18), Qt.AlignCenter, txt)
        if self.X is None: p.end(); return
        if self.X.shape[1] == 2: self._paint_2d(p, w, h)
        elif self.X.shape[1] == 16: self._paint_digits(p, w, h)
        elif self.X.shape[1] == 5: self._paint_hands(p, w, h)
        p.end()

    def _paint_2d(self, p, w, h):
        mg = 30; pw, ph = w - 2 * mg, h - 2 * mg - 10; py0 = mg + 8
        if self._img: p.drawImage(QRectF(mg, py0, pw, ph), self._img)
        xmn, xmx = self._xr; ymn, ymx = self._yr; dx, dy = xmx - xmn, ymx - ymn
        colors = [QColor(50, 200, 255), QColor(255, 80, 80), QColor(80, 255, 80), QColor(255, 200, 50)]
        for i in range(len(self.X)):
            px = mg + (self.X[i, 0] - xmn) / dx * pw; py = py0 + (self.X[i, 1] - ymn) / dy * ph
            c = np.argmax(self.y[i]) if len(self.y[i]) > 1 else int(self.y[i, 0])
            p.setPen(QPen(QColor(0,0,0,100), 0.6)); p.setBrush(QBrush(colors[c % len(colors)]))
            p.drawEllipse(QPointF(px, py), 3, 3)
        p.setPen(QPen(QColor(70,70,90), 1)); p.setBrush(Qt.NoBrush); p.drawRect(QRectF(mg, py0, pw, ph))

    def _paint_digits(self, p, w, h):
        mg = 30; py0 = mg + 8; pw = w - 2 * mg
        if self.nn and self.y.shape[1] > 1 and "Best" in self.label:
            self._draw_confusion_matrix(p, QRectF(mg, py0, pw, h - py0 - 10), DIGIT_LABELS)
        else:
            n_show = min(12, len(self.X)); cols = 4; rows = (n_show + cols - 1) // cols
            cell_w = pw / cols; out = self.nn.forward(self.X[:n_show]) if self.nn else None
            for idx in range(n_show):
                r, c = idx // cols, idx % cols; ox = mg + c * cell_w + 4; oy = py0 + r * (cell_w + 16) + 4
                true_c, pred_c = np.argmax(self.y[idx]), np.argmax(out[idx]) if out is not None else -1
                p.setPen(QPen(QColor(80,255,80) if true_c == pred_c else QColor(255,80,80), 2)); p.setBrush(Qt.NoBrush)
                p.drawRect(QRectF(ox-2, oy-2, cell_w-4, cell_w-4))
                for i in range(16):
                    pr, pc = i // 4, i % 4; v = int(self.X[idx, i] * 255)
                    p.fillRect(QRectF(ox + pc*(cell_w-4)/4, oy + pr*(cell_w-4)/4, (cell_w-4)/4, (cell_w-4)/4), QColor(v, v, v))
                p.setPen(QColor(200,200,220)); p.setFont(QFont("Consolas", 7))
                p.drawText(QPointF(ox, oy + cell_w + 10), f"T:{DIGIT_LABELS[true_c][:2]} P:{DIGIT_LABELS[pred_c][:2] if out is not None else '?'}")

    def _paint_hands(self, p, w, h):
        mg = 30; py0 = mg + 8; pw = w - 2 * mg
        if self.nn and self.y.shape[1] > 1 and "Best" in self.label:
            self._draw_confusion_matrix(p, QRectF(mg, py0, pw, h - py0 - 10), HAND_CLASSES)
        else:
            n_show = min(8, len(self.X)); cols = 4; rows = (n_show + cols - 1) // cols
            cell_w = pw / cols; cell_h = (h - py0 - 20) / rows; out = self.nn.forward(self.X[:n_show]) if self.nn else None
            for idx in range(n_show):
                r, c_idx = idx // cols, idx % cols; ox = mg + c_idx * cell_w + 5; oy = py0 + r * cell_h + 5
                true_c, pred_c = np.argmax(self.y[idx]), np.argmax(out[idx]) if out is not None else -1
                p.setPen(QPen(QColor(80,255,80) if true_c == pred_c else QColor(255,80,80), 2)); p.setBrush(QBrush(QColor(30,30,40)))
                p.drawRect(QRectF(ox-2, oy-2, cell_w-6, cell_h-20))
                bar_h = (cell_h - 30) / 5
                for fi in range(5):
                    v = self.X[idx, fi]
                    p.setPen(Qt.NoPen); p.setBrush(QBrush(QColor(80,170,255) if v > 0.5 else QColor(60,60,80)))
                    p.drawRect(QRectF(ox + fi * (cell_w-6)/5, oy + (1-v)*(cell_h-26), (cell_w-6)/5 - 2, v*(cell_h-26)))
                p.setPen(QColor(200,200,220)); p.setFont(QFont("Consolas", 7))
                p.drawText(QPointF(ox, oy + cell_h - 6), f"T:{HAND_CLASSES[true_c][:2]} P:{HAND_CLASSES[pred_c][:2] if out is not None else '?'}")

    def _draw_confusion_matrix(self, p, rect, labels):
        out = self.nn.forward(self.X); n_classes = len(labels)
        true_labels = np.argmax(self.y, axis=1); pred_labels = np.argmax(out, axis=1)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, pr in zip(true_labels, pred_labels): cm[t, pr] += 1
        
        p.setPen(QColor(180,180,200)); p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(QRectF(rect.x(), rect.y() - 15, rect.width(), 15), Qt.AlignCenter, "Confusion Matrix (Best)")
        
        cell_w = rect.width() / (n_classes + 1); cell_h = rect.height() / (n_classes + 1)
        ox, oy = rect.x() + cell_w, rect.y() + cell_h
        
        for i in range(n_classes):
            p.setPen(QColor(150,150,170)); p.setFont(QFont("Consolas", 7))
            p.drawText(QRectF(ox + i*cell_w, rect.y(), cell_w, cell_h), Qt.AlignCenter, labels[i][:3])
            p.drawText(QRectF(rect.x(), oy + i*cell_h, cell_w, cell_h), Qt.AlignCenter, labels[i][:3])
            
        max_val = cm.max() if cm.max() > 0 else 1
        for i in range(n_classes):
            for j in range(n_classes):
                v = cm[i, j]; intensity = v / max_val
                c = QColor(int(intensity*200), int((1-abs(intensity-0.5)*2)*150), int((1-intensity)*200), 200)
                p.setPen(QPen(QColor(80,80,100), 1)); p.setBrush(QBrush(c))
                p.drawRect(QRectF(ox + j*cell_w, oy + i*cell_h, cell_w, cell_h))
                p.setPen(QColor(255,255,255) if intensity < 0.6 else QColor(0,0,0))
                p.setFont(QFont("Consolas", 9, QFont.Bold)); p.drawText(QRectF(ox + j*cell_w, oy + i*cell_h, cell_w, cell_h), Qt.AlignCenter, str(v))


# ─── Scatter History ──────────────────────────────────────────────────────────

METHOD_COLORS = {
    "he": QColor(80, 200, 120), "xavier": QColor(90, 160, 255), "lecun": QColor(0, 200, 200),
    "uniform": QColor(255, 210, 50), "normal_01": QColor(255, 160, 50), "normal_1": QColor(255, 90, 90),
    "large": QColor(220, 80, 220), "sparse": QColor(170, 130, 255),
}

class HistoryWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.attempts = []; self.best_idx = -1; self.best_acc = -1; self.setMinimumSize(280, 180)

    def add(self, idx, method, acc, loss):
        self.attempts.append((idx, method, acc, loss))
        if acc > self.best_acc: self.best_acc = acc; self.best_idx = len(self.attempts) - 1
        self.update()

    def clear(self): self.attempts = []; self.best_idx = -1; self.best_acc = -1; self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height(); ml, mr, mt, mb = 52, 16, 30, 28
        p.fillRect(self.rect(), QColor(24, 24, 34))
        p.setPen(QColor(190, 190, 210)); p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(QRectF(0, 3, w, 18), Qt.AlignCenter, f"Accuracy per Attempt ({len(self.attempts)})")
        px, py_, pw, ph = ml, mt, w - ml - mr, h - mt - mb
        p.setPen(QPen(QColor(80, 80, 100), 1)); p.drawLine(QPointF(px, py_ + ph), QPointF(px + pw, py_ + ph)); p.drawLine(QPointF(px, py_), QPointF(px, py_ + ph))
        p.setFont(QFont("Consolas", 7))
        for v in np.arange(0, 1.01, 0.25):
            y = py_ + ph - v * ph; p.setPen(QPen(QColor(45, 45, 60), 0.5)); p.drawLine(QPointF(px, y), QPointF(px + pw, y))
            p.setPen(QColor(120, 120, 140)); p.drawText(QRectF(0, y - 6, ml - 4, 12), Qt.AlignRight | Qt.AlignVCenter, f"{v:.0%}")
        p.setPen(QPen(QColor(80, 80, 50), 1, Qt.DashLine)); y50 = py_ + ph * 0.5; p.drawLine(QPointF(px, y50), QPointF(px + pw, y50))
        if not self.attempts: p.end(); return
        n = len(self.attempts)
        for ai, (idx, method, acc, loss) in enumerate(self.attempts):
            x = px + (idx / max(n - 1, 1)) * pw if n > 1 else px + pw / 2; y = py_ + ph - acc * ph
            is_best = (ai == self.best_idx); c = METHOD_COLORS.get(method, QColor(180, 180, 200))
            if is_best:
                grad = QRadialGradient(QPointF(x, y), 12); grad.setColorAt(0, QColor(255, 215, 0, 100)); grad.setColorAt(1, QColor(255, 215, 0, 0))
                p.setPen(Qt.NoPen); p.setBrush(QBrush(grad)); p.drawEllipse(QPointF(x, y), 12, 12)
                p.setPen(QPen(QColor(255, 255, 255), 1.5)); p.setBrush(QBrush(QColor(255, 215, 0))); p.drawEllipse(QPointF(x, y), 5, 5)
            else:
                p.setPen(QPen(QColor(0, 0, 0, 60), 0.5)); p.setBrush(QBrush(c)); p.drawEllipse(QPointF(x, y), 3, 3)
        if self.best_idx >= 0:
            p.setPen(QColor(255, 215, 0)); p.setFont(QFont("Segoe UI", 8, QFont.Bold))
            p.drawText(QRectF(px, py_ + 2, pw, 14), Qt.AlignRight, f"Best: {self.best_acc * 100:.1f}%")
        p.end()


# ─── Histogram ────────────────────────────────────────────────────────────────

class HistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.accs = []; self.best_acc = -1; self.setMinimumSize(280, 140)

    def add(self, acc):
        self.accs.append(acc)
        if acc > self.best_acc: self.best_acc = acc
        self.update()

    def clear(self): self.accs = []; self.best_acc = -1; self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height(); ml, mr, mt, mb = 52, 16, 28, 22
        p.fillRect(self.rect(), QColor(24, 24, 34))
        p.setPen(QColor(190, 190, 210)); p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(QRectF(0, 3, w, 18), Qt.AlignCenter, "Accuracy Distribution")
        px, py_, pw, ph = ml, mt, w - ml - mr, h - mt - mb
        if not self.accs: p.end(); return
        n_bins = min(20, max(5, len(self.accs) // 3)); bins = np.linspace(0, 1, n_bins + 1)
        counts, _ = np.histogram(self.accs, bins=bins); max_c = max(counts.max(), 1); bw = pw / n_bins
        for i in range(n_bins):
            x = px + i * bw; bh = counts[i] / max_c * ph; mid_acc = (bins[i] + bins[i + 1]) / 2
            c = QColor(int((1 - mid_acc) * 200 + 40), int(mid_acc * 200 + 40), 60, 180)
            p.setPen(QPen(QColor(200, 200, 220, 60), 0.5)); p.setBrush(QBrush(c)); p.drawRect(QRectF(x + 1, py_ + ph - bh, bw - 2, bh))
        if self.best_acc >= 0:
            bx = px + self.best_acc * pw; p.setPen(QPen(QColor(255, 215, 0), 2, Qt.DashLine)); p.drawLine(QPointF(bx, py_), QPointF(bx, py_ + ph))
            p.setPen(QColor(255, 215, 0)); p.setFont(QFont("Consolas", 7)); p.drawText(QPointF(bx + 3, py_ + 10), f"Best {self.best_acc:.1%}")
        p.end()


# ─── Main Window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    METHODS = ["he", "xavier", "lecun", "uniform", "normal_01", "normal_1", "large", "sparse"]
    VIDEO_FPS = 30

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Random Weight Search + Digits/Hands/Custom + Auto MP4")
        self.setMinimumSize(1100, 720); self.resize(1400, 900)

        self.nn = None; self.best_nn = None; self.X = self.y = None
        self.running = False; self.current_iter = 0; self.total_iter = 100
        self.best_acc = -1; self.current_acc = 0; self.current_loss = 0
        
        # Export State
        self.is_exporting = False; self.video_writer = None; self.video_filename = ""

        self._build_ui(); self._build_timers(); self._full_reset()

    def _build_ui(self):
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(34, 34, 46)); pal.setColor(QPalette.WindowText, QColor(200, 200, 220))
        pal.setColor(QPalette.Base, QColor(30, 30, 42)); pal.setColor(QPalette.Text, QColor(200, 200, 220))
        pal.setColor(QPalette.Button, QColor(48, 48, 62)); pal.setColor(QPalette.ButtonText, QColor(200, 200, 220))
        self.setPalette(pal)
        self.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #555; border-radius: 5px; margin-top: 14px; padding-top: 16px; color: #c8c8dc; }
            QGroupBox::title { subcontrol-origin: margin; padding: 0 6px; }
            QComboBox, QSpinBox, QDoubleSpinBox { padding: 3px; border: 1px solid #555; border-radius: 3px; background: #38384c; color: #ddd; }
            QProgressBar { border: 1px solid #555; border-radius: 3px; text-align: center; background: #2a2a3a; color: #ddd; font-weight: bold; }
            QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #6a50a0, stop:1 #4090c0); border-radius: 2px; }
            QCheckBox { color: #ccc; } QLabel { color: #bbb; }
            QDialog { background-color: #2a2a3a; color: #ddd; } QRadioButton { color: #ccc; }
        """)

        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(6, 6, 6, 6); root.setSpacing(6)

        # LEFT
        left = QWidget(); left.setFixedWidth(255); lv = QVBoxLayout(left); lv.setContentsMargins(0,0,0,0); lv.setSpacing(4)

        g = QGroupBox("Dataset"); gl = QVBoxLayout(g); gl.addWidget(QLabel("Type:"))
        self.dataset_cb = QComboBox()
        self.dataset_cb.addItems(["xor", "circle", "spiral", "moons", "gaussian", "digits_4x4", "hand_signs", "custom_draw"])
        self.dataset_cb.currentTextChanged.connect(self._on_data_changed); gl.addWidget(self.dataset_cb)
        self.draw_btn = QPushButton("✏️ Open Drawing Canvas")
        self.draw_btn.setStyleSheet("QPushButton{background:#2a6090;color:#fff;padding:5px;border-radius:3px}QPushButton:hover{background:#3a70a0}")
        self.draw_btn.clicked.connect(self._open_draw_dialog); gl.addWidget(self.draw_btn)
        hl = QHBoxLayout(); hl.addWidget(QLabel("Samples:"))
        self.nspin = QSpinBox(); self.nspin.setRange(50, 2000); self.nspin.setValue(400); self.nspin.setSingleStep(50)
        self.nspin.valueChanged.connect(self._on_data_changed); hl.addWidget(self.nspin); gl.addLayout(hl); lv.addWidget(g)

        g = QGroupBox("Architecture"); gl = QGridLayout(g)
        self.in_label = QLabel("Input: 2"); gl.addWidget(self.in_label, 0, 0, 1, 2)
        for r, (label, default) in enumerate([("Hidden 1:", 8), ("Hidden 2:", 8), ("Hidden 3:", 0)], 1):
            gl.addWidget(QLabel(label), r, 0); sp = QSpinBox(); sp.setRange(0, 64); sp.setValue(default)
            sp.valueChanged.connect(self._on_arch_changed); gl.addWidget(sp, r, 1); setattr(self, f"h{r}_spin", sp)
        self.out_label = QLabel("Output: 1"); gl.addWidget(self.out_label, 4, 0, 1, 2); lv.addWidget(g)

        g = QGroupBox("Randomization"); gl = QVBoxLayout(g); gl.addWidget(QLabel("Init method:"))
        self.method_cb = QComboBox(); self.method_cb.addItems(self.METHODS); gl.addWidget(self.method_cb)
        self.compare_cb = QCheckBox("Cycle ALL methods"); gl.addWidget(self.compare_cb); lv.addWidget(g)

        g = QGroupBox("Interactive Search"); gl = QVBoxLayout(g)
        hl = QHBoxLayout(); hl.addWidget(QLabel("Iterations:"))
        self.iter_spin = QSpinBox(); self.iter_spin.setRange(10, 10000); self.iter_spin.setValue(200); self.iter_spin.setSingleStep(50)
        hl.addWidget(self.iter_spin); gl.addLayout(hl)
        hl = QHBoxLayout(); hl.addWidget(QLabel("Speed:"))
        self.speed_spin = QSpinBox(); self.speed_spin.setRange(1, 200); self.speed_spin.setValue(3); self.speed_spin.setSuffix(" / tick")
        hl.addWidget(self.speed_spin); gl.addLayout(hl)

        self.start_btn = QPushButton("🎲  Start Search")
        self.start_btn.setStyleSheet("QPushButton{background:#2a7090;color:#fff;padding:8px;border-radius:4px;font-weight:bold;font-size:12px}QPushButton:hover{background:#3a80a0}")
        self.start_btn.clicked.connect(self._toggle_run); gl.addWidget(self.start_btn)
        self.step_btn = QPushButton("→  Single Step")
        self.step_btn.setStyleSheet("QPushButton{background:#505068;color:#ddd;padding:5px;border-radius:3px}QPushButton:hover{background:#606080}")
        self.step_btn.clicked.connect(self._single_step); gl.addWidget(self.step_btn)
        self.reset_btn = QPushButton("↺  Full Reset")
        self.reset_btn.setStyleSheet("QPushButton{background:#804040;color:#fff;padding:5px;border-radius:3px}QPushButton:hover{background:#905050}")
        self.reset_btn.clicked.connect(self._full_reset); gl.addWidget(self.reset_btn)
        self.progress = QProgressBar(); self.progress.setVisible(False); gl.addWidget(self.progress); lv.addWidget(g)

        g = QGroupBox("Live Stats"); gl = QVBoxLayout(g)
        self.cur_label = QLabel("Current:  —"); self.cur_label.setStyleSheet("color:#aaa;font-size:11px"); gl.addWidget(self.cur_label)
        self.best_label = QLabel("Best:  —"); self.best_label.setStyleSheet("color:#ffd700;font-size:12px;font-weight:bold"); gl.addWidget(self.best_label)
        self.iter_label = QLabel("Iteration: 0"); self.iter_label.setStyleSheet("color:#999;font-size:10px"); gl.addWidget(self.iter_label)
        lv.addWidget(g)

        # AUTO EXPORT VIDEO
        g = QGroupBox("Auto MP4 Export (Full Run)"); gl = QVBoxLayout(g)
        if not CV2_AVAILABLE:
            gl.addWidget(QLabel("⚠ pip install opencv-python\nto enable recording"))
            self.export_btn = QPushButton("Export (Missing CV2)"); self.export_btn.setEnabled(False)
        else:
            gl.addWidget(QLabel("Automatically runs all iterations\nand saves 1080p MP4 to /output/"))
            self.export_btn = QPushButton("🎬  Export Full Run to MP4")
        self.export_btn.setStyleSheet("QPushButton{background:#a02020;color:#fff;padding:8px;border-radius:4px;font-weight:bold;font-size:12px}QPushButton:hover{background:#c03030}")
        self.export_btn.clicked.connect(self._start_auto_export); gl.addWidget(self.export_btn)
        self.export_status = QLabel(""); self.export_status.setStyleSheet("color:#ff8888;font-size:10px"); gl.addWidget(self.export_status)
        lv.addWidget(g); lv.addStretch(); root.addWidget(left)

        # RIGHT
        splitter = QSplitter(Qt.Vertical)
        row1 = QSplitter(Qt.Horizontal)
        self.viz_cur = DataVizWidget(); self.viz_best = DataVizWidget()
        row1.addWidget(self.viz_cur); row1.addWidget(self.viz_best); row1.setSizes([500, 500])
        row2 = QSplitter(Qt.Horizontal)
        self.history = HistoryWidget(); self.histogram = HistogramWidget()
        row2.addWidget(self.history); row2.addWidget(self.histogram); row2.setSizes([500, 300])
        self.net_viz = NetworkVizWidget()
        splitter.addWidget(row1); splitter.addWidget(row2); splitter.addWidget(self.net_viz)
        splitter.setSizes([320, 220, 280]); root.addWidget(splitter, 1)
        self.statusBar().setStyleSheet("color:#aaa;")

    def _build_timers(self):
        self.timer = QTimer(); self.timer.timeout.connect(self._tick); self.timer.setInterval(25)
        self.export_timer = QTimer(); self.export_timer.timeout.connect(self._export_tick); self.export_timer.setInterval(5)

    # ── helpers ──

    def _layer_sizes(self):
        ds = self.dataset_cb.currentText()
        n_in, n_out = DataGenerator.get_shape(ds) if ds != "custom_draw" else (self.X.shape[1] if self.X is not None else 2, self.y.shape[1] if self.y is not None else 1)
        layers = [n_in]
        for sp in (self.h1_spin, self.h2_spin, self.h3_spin):
            if sp.value() > 0: layers.append(sp.value())
        layers.append(n_out); return layers

    def _get_method(self):
        if self.compare_cb.isChecked(): return self.METHODS[self.current_iter % len(self.METHODS)]
        return self.method_cb.currentText()

    def _update_arch_labels(self):
        layers = self._layer_sizes(); self.in_label.setText(f"Input: {layers[0]}"); self.out_label.setText(f"Output: {layers[-1]}")

    def _full_reset(self):
        self._stop(); self._stop_export()
        self.current_iter = 0; self.best_acc = -1; self.best_nn = None
        ds = self.dataset_cb.currentText()
        if ds == "custom_draw" and (self.X is None or len(self.X) == 0):
            self.X, self.y = DataGenerator.generate("gaussian", self.nspin.value())
        elif ds != "custom_draw":
            self.X, self.y = DataGenerator.generate(ds, self.nspin.value())
        self._update_arch_labels()
        self.nn = NeuralNetwork(self._layer_sizes()); self.nn.randomize(self.method_cb.currentText())
        acc, loss = self.nn.evaluate(self.X, self.y); self.current_acc = acc; self.current_loss = loss
        self.viz_cur.set_data(self.X, self.y); self.viz_best.set_data(self.X, self.y)
        self.viz_cur.update_network(self.nn, "Current Candidate", acc)
        self.viz_best.update_network(None, "Best Found (none yet)")
        self.net_viz.set_network(self.nn, "Current Candidate")
        self.history.clear(); self.histogram.clear()
        self.cur_label.setText(f"Current:  Acc {acc:.1%}  Loss {loss:.4f}")
        self.best_label.setText("Best:  —"); self.iter_label.setText("Iteration: 0")

    def _refresh_current(self, method):
        acc, loss = self.nn.evaluate(self.X, self.y); self.current_acc = acc; self.current_loss = loss
        is_new_best = acc > self.best_acc
        if is_new_best:
            self.best_acc = acc
            if self.best_nn is None: self.best_nn = NeuralNetwork(self._layer_sizes())
            self.best_nn.copy_from(self.nn)
            self.viz_best.update_network(self.best_nn, f"★ Best Found", acc)
        self.viz_cur.update_network(self.nn, f"#{self.current_iter + 1} ({method})", acc)
        self.net_viz.set_network(self.nn, f"Candidate #{self.current_iter + 1}  [{method}]", is_best=is_new_best)
        self.history.add(self.current_iter, method, acc, loss); self.histogram.add(acc)
        self.cur_label.setText(f"Current:  Acc {acc:.1%}  Loss {loss:.4f}")
        if self.best_nn: self.best_label.setText(f"Best:  Acc {self.best_acc:.1%}  (iter #{self.history.attempts[self.history.best_idx][0] + 1})")
        self.iter_label.setText(f"Iteration: {self.current_iter + 1} / {self.total_iter}")
        if is_new_best: self.statusBar().showMessage(f"🎉 NEW BEST at #{self.current_iter + 1}!  Acc = {acc:.1%}  ({method})")
        else: self.statusBar().showMessage(f"#{self.current_iter + 1}  Acc={acc:.1%}  Best={self.best_acc:.1%}  ({method})")

    def _open_draw_dialog(self):
        dlg = DrawingDialog(self)
        if dlg.exec() == QDialog.Accepted:
            X, y = dlg.get_data()
            if X is not None:
                self.dataset_cb.setCurrentText("custom_draw")
                self.X, self.y = X, y; self._full_reset()

    # ── Interactive Simulation Controls ──

    def _on_data_changed(self, *_): self._full_reset()
    def _on_arch_changed(self, *_): self._full_reset()

    def _toggle_run(self):
        if self.running: self._stop()
        else: self._start()

    def _start(self):
        self.running = True; self.total_iter = self.iter_spin.value()
        self.progress.setVisible(True); self.progress.setRange(0, self.total_iter); self.progress.setValue(self.current_iter)
        self.start_btn.setText("⏹  Stop")
        self.start_btn.setStyleSheet("QPushButton{background:#c09020;color:#fff;padding:8px;border-radius:4px;font-weight:bold;font-size:12px}QPushButton:hover{background:#d0a030}")
        self.timer.start()

    def _stop(self):
        self.timer.stop(); self.running = False
        self.start_btn.setText("🎲  Start Search")
        self.start_btn.setStyleSheet("QPushButton{background:#2a7090;color:#fff;padding:8px;border-radius:4px;font-weight:bold;font-size:12px}QPushButton:hover{background:#3a80a0}")
        if self.current_iter >= self.total_iter:
            self.progress.setVisible(False)
            self.statusBar().showMessage(f"Search complete!  Best accuracy: {self.best_acc:.1%}")

    def _tick(self):
        if not self.running: return
        for _ in range(self.speed_spin.value()):
            if self.current_iter >= self.total_iter: self._stop(); return
            method = self._get_method(); self.nn = NeuralNetwork(self._layer_sizes()); self.nn.randomize(method)
            self._refresh_current(method); self.current_iter += 1; self.progress.setValue(self.current_iter)

    def _single_step(self):
        if self.running: return
        self.total_iter = self.iter_spin.value(); method = self._get_method()
        self.nn = NeuralNetwork(self._layer_sizes()); self.nn.randomize(method)
        self._refresh_current(method); self.current_iter += 1

    # ── Automated Video Export Engine ──

    def _start_auto_export(self):
        if not CV2_AVAILABLE or self.is_exporting: return
        os.makedirs("output", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"output/random_nn_{self.dataset_cb.currentText()}_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(fname, fourcc, self.VIDEO_FPS, (1920, 1080))
        if not self.video_writer.isOpened():
            self.statusBar().showMessage("Error: Could not open video writer."); self.video_writer = None; return

        self.is_exporting = True; self.video_filename = fname
        self.export_btn.setText("⏹  Exporting...")
        self.export_btn.setEnabled(False); self.export_status.setText(f"Writing to {fname}")
        
        # Reset simulation for the export
        self.current_iter = 0; self.best_acc = -1; self.best_nn = None
        self.X, self.y = DataGenerator.generate(self.dataset_cb.currentText(), self.nspin.value())
        self._update_arch_labels()
        self.nn = NeuralNetwork(self._layer_sizes()); self.nn.randomize(self.method_cb.currentText())
        self.viz_cur.set_data(self.X, self.y); self.viz_best.set_data(self.X, self.y)
        self.history.clear(); self.histogram.clear()
        
        self.total_iter = self.iter_spin.value()
        self.progress.setVisible(True); self.progress.setRange(0, self.total_iter); self.progress.setValue(0)

        # Write intro frames
        for _ in range(self.VIDEO_FPS * 2): self._capture_video_frame(intro=True)
        
        self.export_timer.start()

    def _export_tick(self):
        if not self.is_exporting: return
        
        # Process 1 iteration per tick
        method = self._get_method()
        self.nn = NeuralNetwork(self._layer_sizes()); self.nn.randomize(method)
        self._refresh_current(method)
        
        # Write frames for this iteration (e.g., 3 frames = 0.1s at 30fps)
        for _ in range(3): self._capture_video_frame(is_new_best=(self.current_acc >= self.best_acc))
        
        self.current_iter += 1; self.progress.setValue(self.current_iter)
        
        if self.current_iter >= self.total_iter: self._stop_export()

    def _stop_export(self):
        if not self.is_exporting: return
        self.export_timer.stop()
        # Write outro frames
        for _ in range(self.VIDEO_FPS * 3): self._capture_video_frame(outro=True)
        
        self.is_exporting = False
        if self.video_writer: self.video_writer.release(); self.video_writer = None
        self.export_btn.setText("🎬  Export Full Run to MP4"); self.export_btn.setEnabled(True)
        self.export_status.setText(f"✅ Saved: {self.video_filename}")
        self.progress.setVisible(False)
        self.statusBar().showMessage(f"MP4 Export Complete: {self.video_filename}")

    # ── Video Compositing Engine ──

    def _compose_video_frame(self, intro=False, outro=False):
        canvas = QImage(1920, 1080, QImage.Format_RGB32); canvas.fill(QColor(18, 18, 28))
        p = QPainter(canvas); p.setRenderHint(QPainter.Antialiasing); p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.setPen(QColor(255, 255, 255)); p.setFont(QFont("Segoe UI", 36, QFont.Bold))
        p.drawText(QRectF(40, 20, 1500, 50), Qt.AlignLeft, "Neural Network Random Weight Search")
        p.setFont(QFont("Segoe UI", 20)); p.setPen(QColor(180, 180, 200))
        ds = self.dataset_cb.currentText().upper(); arch = " x ".join(map(str, self._layer_sizes()))
        p.drawText(QRectF(40, 75, 1500, 30), Qt.AlignLeft, f"Dataset: {ds}  |  Architecture: [{arch}]  |  Init: {self._get_method().upper()}")
        
        if intro:
            p.fillRect(0, 0, 1920, 1080, QColor(0, 0, 0, 150))
            p.setPen(QColor(255, 255, 255)); p.setFont(QFont("Segoe UI", 64, QFont.Bold))
            p.drawText(QRectF(0, 350, 1920, 100), Qt.AlignCenter, "Random Weight Search")
            p.setFont(QFont("Segoe UI", 32)); p.setPen(QColor(200, 200, 220))
            p.drawText(QRectF(0, 470, 1920, 60), Qt.AlignCenter, f"Can purely random weights solve {ds}?")
            p.drawText(QRectF(0, 540, 1920, 60), Qt.AlignCenter, "No training. Just random initialization and evaluation.")
            p.end(); return canvas
        if outro:
            p.fillRect(0, 0, 1920, 1080, QColor(0, 0, 0, 150))
            p.setPen(QColor(255, 215, 0)); p.setFont(QFont("Segoe UI", 64, QFont.Bold))
            p.drawText(QRectF(0, 350, 1920, 100), Qt.AlignCenter, "Search Complete!")
            p.setFont(QFont("Segoe UI", 36)); p.setPen(QColor(255, 255, 255))
            p.drawText(QRectF(0, 470, 1920, 60), Qt.AlignCenter, f"Best Accuracy Found: {self.best_acc:.1%}")
            p.end(); return canvas

        p.setFont(QFont("Consolas", 22, QFont.Bold)); p.setPen(QColor(255, 255, 255))
        p.drawText(QRectF(1400, 20, 500, 35), Qt.AlignRight, f"Iteration: {self.current_iter + 1}/{self.total_iter}")
        p.setPen(QColor(255, 215, 0)); p.drawText(QRectF(1400, 55, 500, 35), Qt.AlignRight, f"Best Acc: {self.best_acc:.1%}")

        pix_cur = self.viz_cur.grab(); pix_best = self.viz_best.grab()
        pix_net = self.net_viz.grab(); pix_scat = self.history.grab(); pix_hist = self.histogram.grab()
        def draw_scaled(painter, pixmap, target_rect):
            scaled = pixmap.scaled(target_rect.width(), target_rect.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = target_rect.x() + (target_rect.width() - scaled.width()) / 2; y = target_rect.y() + (target_rect.height() - scaled.height()) / 2
            painter.drawPixmap(QPointF(x, y), scaled)
        draw_scaled(p, pix_cur, QRectF(10, 110, 950, 480)); draw_scaled(p, pix_best, QRectF(960, 110, 950, 480))
        draw_scaled(p, pix_net, QRectF(10, 600, 630, 480)); draw_scaled(p, pix_scat, QRectF(640, 600, 640, 480)); draw_scaled(p, pix_hist, QRectF(1280, 600, 640, 480))
        p.setPen(QPen(QColor(60, 60, 80), 2)); p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(10, 110, 950, 480)); p.drawRect(QRectF(960, 110, 950, 480))
        p.drawRect(QRectF(10, 600, 630, 480)); p.drawRect(QRectF(640, 600, 640, 480)); p.drawRect(QRectF(1280, 600, 640, 480))
        p.end(); return canvas

    def _capture_video_frame(self, intro=False, outro=False, is_new_best=False):
        if not self.is_exporting or not self.video_writer: return
        canvas = self._compose_video_frame(intro=intro, outro=outro)
        canvas = canvas.convertToFormat(QImage.Format_RGB888)
        width = canvas.width()
        height = canvas.height()
        
        # FIX for PySide6 memoryview conversion
        ptr = canvas.bits()
        arr = np.frombuffer(ptr, dtype=np.uint8)
        
        # Handle potential stride/padding in QImage
        bpl = canvas.bytesPerLine()
        if bpl == width * 3:
            arr = arr.reshape(height, width, 3)
        else:
            # If there is padding, slice it out
            arr = arr.reshape(height, bpl)
            arr = arr[:, :width * 3].reshape(height, width, 3)
            
        frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())