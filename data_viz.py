"""
Universal data / decision-boundary visualization widget.
Handles 2D scatter, 4×4 digit grids, hand-sign bars, and confusion matrices.
"""

import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QImage
)

# ─── Label Constants ──────────────────────────────────────────────────────────

HAND_LABELS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
DIGIT_LABELS = ["H-Line", "V-Line", "Diag"]
HAND_CLASSES = ["Fist", "Peace", "Point", "Open"]


class DataVizWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.nn = None
        self.X = self.y = None
        self._img = None
        self._xr = self._yr = (-1, 1)
        self.label = ""
        self.acc = None
        self.resolution = 60
        self.setMinimumSize(260, 260)

    def set_data(self, X, y):
        self.X, self.y = X, y

    def update_network(self, nn, label="", acc=None):
        self.nn = nn
        self.label = label
        self.acc = acc
        self._rebuild()

    def _rebuild(self):
        if self.nn is None or self.X is None:
            self._img = None
            self.update()
            return
        if self.X.shape[1] == 2:
            xmn, xmx = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
            ymn, ymx = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
            res = self.resolution
            xx, yy = np.meshgrid(np.linspace(xmn, xmx, res), np.linspace(ymn, ymx, res))
            Z = self.nn.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(res, res, -1)
            self._xr, self._yr = (xmn, xmx), (ymn, ymx)

            img = QImage(res, res, QImage.Format_RGB32)
            n_out = Z.shape[2]
            colors = [
                QColor(50, 200, 255), QColor(255, 80, 80),
                QColor(80, 255, 80), QColor(255, 200, 50), QColor(200, 80, 255),
            ]
            for i in range(res):
                for j in range(res):
                    if n_out == 1:
                        v = Z[i, j, 0]
                        img.setPixelColor(
                            j, i,
                            QColor(
                                int(np.clip(v * 200 + 25, 0, 255)),
                                int(np.clip(min(v, 1 - v) * 4 * 50, 0, 255)),
                                int(np.clip((1 - v) * 200 + 25, 0, 255)),
                            ),
                        )
                    else:
                        mx_idx = np.argmax(Z[i, j])
                        img.setPixelColor(j, i, colors[mx_idx % len(colors)])
            self._img = img
        else:
            self._img = None
        self.update()

    # ── Paint Dispatch ────────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(24, 24, 34))

        txt = self.label
        if self.acc is not None:
            txt += f"  Acc: {self.acc * 100:.1f}%"
        p.setPen(QColor(190, 190, 210))
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(QRectF(0, 3, w, 18), Qt.AlignCenter, txt)

        if self.X is None:
            p.end()
            return

        if self.X.shape[1] == 2:
            self._paint_2d(p, w, h)
        elif self.X.shape[1] == 16:
            self._paint_digits(p, w, h)
        elif self.X.shape[1] == 5:
            self._paint_hands(p, w, h)

        p.end()

    # ── 2D Scatter + Decision Boundary ────────────────────────────────────────

    def _paint_2d(self, p, w, h):
        mg = 30
        pw, ph = w - 2 * mg, h - 2 * mg - 10
        py0 = mg + 8

        if self._img:
            p.drawImage(QRectF(mg, py0, pw, ph), self._img)

        xmn, xmx = self._xr
        ymn, ymx = self._yr
        dx, dy = xmx - xmn, ymx - ymn
        colors = [QColor(50, 200, 255), QColor(255, 80, 80),
                  QColor(80, 255, 80), QColor(255, 200, 50)]

        for i in range(len(self.X)):
            px = mg + (self.X[i, 0] - xmn) / dx * pw
            py = py0 + (self.X[i, 1] - ymn) / dy * ph
            c = np.argmax(self.y[i]) if len(self.y[i]) > 1 else int(self.y[i, 0])
            p.setPen(QPen(QColor(0, 0, 0, 100), 0.6))
            p.setBrush(QBrush(colors[c % len(colors)]))
            p.drawEllipse(QPointF(px, py), 3, 3)

        p.setPen(QPen(QColor(70, 70, 90), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(mg, py0, pw, ph))

    # ── 4×4 Digit Grids ──────────────────────────────────────────────────────

    def _paint_digits(self, p, w, h):
        mg = 30
        py0 = mg + 8
        pw = w - 2 * mg

        if self.nn and self.y.shape[1] > 1 and "Best" in self.label:
            self._draw_confusion_matrix(p, QRectF(mg, py0, pw, h - py0 - 10), DIGIT_LABELS)
        else:
            n_show = min(12, len(self.X))
            cols = 4
            rows = (n_show + cols - 1) // cols
            cell_w = pw / cols
            out = self.nn.forward(self.X[:n_show]) if self.nn else None

            for idx in range(n_show):
                r, c = idx // cols, idx % cols
                ox = mg + c * cell_w + 4
                oy = py0 + r * (cell_w + 16) + 4
                true_c = np.argmax(self.y[idx])
                pred_c = np.argmax(out[idx]) if out is not None else -1

                p.setPen(QPen(QColor(80, 255, 80) if true_c == pred_c else QColor(255, 80, 80), 2))
                p.setBrush(Qt.NoBrush)
                p.drawRect(QRectF(ox - 2, oy - 2, cell_w - 4, cell_w - 4))

                for i in range(16):
                    pr, pc = i // 4, i % 4
                    v = int(self.X[idx, i] * 255)
                    p.fillRect(
                        QRectF(
                            ox + pc * (cell_w - 4) / 4,
                            oy + pr * (cell_w - 4) / 4,
                            (cell_w - 4) / 4,
                            (cell_w - 4) / 4,
                        ),
                        QColor(v, v, v),
                    )

                p.setPen(QColor(200, 200, 220))
                p.setFont(QFont("Consolas", 7))
                p.drawText(
                    QPointF(ox, oy + cell_w + 10),
                    f"T:{DIGIT_LABELS[true_c][:2]} P:{DIGIT_LABELS[pred_c][:2] if out is not None else '?'}",
                )

    # ── Hand-Sign Bar Charts ─────────────────────────────────────────────────

    def _paint_hands(self, p, w, h):
        mg = 30
        py0 = mg + 8
        pw = w - 2 * mg

        if self.nn and self.y.shape[1] > 1 and "Best" in self.label:
            self._draw_confusion_matrix(p, QRectF(mg, py0, pw, h - py0 - 10), HAND_CLASSES)
        else:
            n_show = min(8, len(self.X))
            cols = 4
            rows = (n_show + cols - 1) // cols
            cell_w = pw / cols
            cell_h = (h - py0 - 20) / rows
            out = self.nn.forward(self.X[:n_show]) if self.nn else None

            for idx in range(n_show):
                r, c_idx = idx // cols, idx % cols
                ox = mg + c_idx * cell_w + 5
                oy = py0 + r * cell_h + 5
                true_c = np.argmax(self.y[idx])
                pred_c = np.argmax(out[idx]) if out is not None else -1

                p.setPen(QPen(QColor(80, 255, 80) if true_c == pred_c else QColor(255, 80, 80), 2))
                p.setBrush(QBrush(QColor(30, 30, 40)))
                p.drawRect(QRectF(ox - 2, oy - 2, cell_w - 6, cell_h - 20))

                bar_h = (cell_h - 30) / 5
                for fi in range(5):
                    v = self.X[idx, fi]
                    p.setPen(Qt.NoPen)
                    p.setBrush(QBrush(QColor(80, 170, 255) if v > 0.5 else QColor(60, 60, 80)))
                    p.drawRect(
                        QRectF(
                            ox + fi * (cell_w - 6) / 5,
                            oy + (1 - v) * (cell_h - 26),
                            (cell_w - 6) / 5 - 2,
                            v * (cell_h - 26),
                        )
                    )

                p.setPen(QColor(200, 200, 220))
                p.setFont(QFont("Consolas", 7))
                p.drawText(
                    QPointF(ox, oy + cell_h - 6),
                    f"T:{HAND_CLASSES[true_c][:2]} P:{HAND_CLASSES[pred_c][:2] if out is not None else '?'}",
                )

    # ── Confusion Matrix ─────────────────────────────────────────────────────

    def _draw_confusion_matrix(self, p, rect, labels):
        out = self.nn.forward(self.X)
        n_classes = len(labels)
        true_labels = np.argmax(self.y, axis=1)
        pred_labels = np.argmax(out, axis=1)

        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, pr in zip(true_labels, pred_labels):
            cm[t, pr] += 1

        p.setPen(QColor(180, 180, 200))
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(QRectF(rect.x(), rect.y() - 15, rect.width(), 15),
                   Qt.AlignCenter, "Confusion Matrix (Best)")

        cell_w = rect.width() / (n_classes + 1)
        cell_h = rect.height() / (n_classes + 1)
        ox, oy = rect.x() + cell_w, rect.y() + cell_h

        # Row/column headers
        for i in range(n_classes):
            p.setPen(QColor(150, 150, 170))
            p.setFont(QFont("Consolas", 7))
            p.drawText(QRectF(ox + i * cell_w, rect.y(), cell_w, cell_h),
                       Qt.AlignCenter, labels[i][:3])
            p.drawText(QRectF(rect.x(), oy + i * cell_h, cell_w, cell_h),
                       Qt.AlignCenter, labels[i][:3])

        # Cells
        max_val = cm.max() if cm.max() > 0 else 1
        for i in range(n_classes):
            for j in range(n_classes):
                v = cm[i, j]
                intensity = v / max_val
                c = QColor(
                    int(intensity * 200),
                    int((1 - abs(intensity - 0.5) * 2) * 150),
                    int((1 - intensity) * 200),
                    200,
                )
                p.setPen(QPen(QColor(80, 80, 100), 1))
                p.setBrush(QBrush(c))
                p.drawRect(QRectF(ox + j * cell_w, oy + i * cell_h, cell_w, cell_h))

                p.setPen(QColor(255, 255, 255) if intensity < 0.6 else QColor(0, 0, 0))
                p.setFont(QFont("Consolas", 9, QFont.Bold))
                p.drawText(QRectF(ox + j * cell_w, oy + i * cell_h, cell_w, cell_h),
                           Qt.AlignCenter, str(v))