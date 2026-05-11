"""
MP4 video export engine — composites all widgets into 1080p frames via OpenCV.
"""

import os
import datetime
import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import (
    QImage, QPainter, QFont, QColor, QPen, QRectF as GuiRectF
)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class VideoExporter:
    """Handles compositing and writing of MP4 video frames."""

    VIDEO_FPS = 30
    FRAMES_PER_ITERATION = 3
    INTRO_SECONDS = 2
    OUTRO_SECONDS = 3
    WIDTH = 1920
    HEIGHT = 1080

    def __init__(self, main_window):
        """
        Parameters
        ----------
        main_window : MainWindow
            Reference to the main window for accessing widgets and state.
        """
        self.mw = main_window
        self.is_exporting = False
        self.video_writer = None
        self.video_filename = ""

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, dataset_name, total_iter):
        """Initialize the video writer and write intro frames.

        Returns True on success, False on failure.
        """
        if not CV2_AVAILABLE:
            return False

        os.makedirs("output", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"output/random_nn_{dataset_name}_{timestamp}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            fname, fourcc, self.VIDEO_FPS, (self.WIDTH, self.HEIGHT)
        )
        if not self.video_writer.isOpened():
            self.video_writer = None
            return False

        self.is_exporting = True
        self.video_filename = fname

        # Intro frames
        for _ in range(self.VIDEO_FPS * self.INTRO_SECONDS):
            self._write_frame(self._compose_frame(intro=True))

        return True

    def write_iteration_frames(self, is_new_best=False):
        """Write frames for the current simulation iteration."""
        if not self.is_exporting or not self.video_writer:
            return
        for _ in range(self.FRAMES_PER_ITERATION):
            self._write_frame(self._compose_frame())

    def finish(self):
        """Write outro frames, release writer, and return the filename."""
        if not self.is_exporting:
            return ""

        # Outro frames
        for _ in range(self.VIDEO_FPS * self.OUTRO_SECONDS):
            self._write_frame(self._compose_frame(outro=True))

        self.is_exporting = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        return self.video_filename

    # ── Frame Compositing ─────────────────────────────────────────────────────

    def _compose_frame(self, intro=False, outro=False):
        """Compose a single 1080p video frame as a QImage."""
        canvas = QImage(self.WIDTH, self.HEIGHT, QImage.Format_RGB32)
        canvas.fill(QColor(18, 18, 28))
        p = QPainter(canvas)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        ds = self.mw.dataset_cb.currentText().upper()
        arch = " x ".join(map(str, self.mw._layer_sizes()))
        method = self.mw._get_method().upper()

        # ── Intro ─────────────────────────────────────────────────────────
        if intro:
            p.fillRect(0, 0, self.WIDTH, self.HEIGHT, QColor(0, 0, 0, 150))
            p.setPen(QColor(255, 255, 255))
            p.setFont(QFont("Segoe UI", 64, QFont.Bold))
            p.drawText(QRectF(0, 350, self.WIDTH, 100),
                       Qt.AlignCenter, "Random Weight Search")
            p.setFont(QFont("Segoe UI", 32))
            p.setPen(QColor(200, 200, 220))
            p.drawText(QRectF(0, 470, self.WIDTH, 60),
                       Qt.AlignCenter, f"Can purely random weights solve {ds}?")
            p.drawText(QRectF(0, 540, self.WIDTH, 60),
                       Qt.AlignCenter, "No training. Just random initialization and evaluation.")
            p.end()
            return canvas

        # ── Outro ─────────────────────────────────────────────────────────
        if outro:
            p.fillRect(0, 0, self.WIDTH, self.HEIGHT, QColor(0, 0, 0, 150))
            p.setPen(QColor(255, 215, 0))
            p.setFont(QFont("Segoe UI", 64, QFont.Bold))
            p.drawText(QRectF(0, 350, self.WIDTH, 100),
                       Qt.AlignCenter, "Search Complete!")
            p.setFont(QFont("Segoe UI", 36))
            p.setPen(QColor(255, 255, 255))
            p.drawText(QRectF(0, 470, self.WIDTH, 60),
                       Qt.AlignCenter, f"Best Accuracy Found: {self.mw.best_acc:.1%}")
            p.end()
            return canvas

        # ── Normal frame ──────────────────────────────────────────────────
        # Header
        p.setPen(QColor(255, 255, 255))
        p.setFont(QFont("Segoe UI", 36, QFont.Bold))
        p.drawText(QRectF(40, 20, 1500, 50), Qt.AlignLeft,
                   "Neural Network Random Weight Search")
        p.setFont(QFont("Segoe UI", 20))
        p.setPen(QColor(180, 180, 200))
        p.drawText(QRectF(40, 75, 1500, 30), Qt.AlignLeft,
                   f"Dataset: {ds}  |  Architecture: [{arch}]  |  Init: {method}")

        # Iteration / best overlay
        p.setFont(QFont("Consolas", 22, QFont.Bold))
        p.setPen(QColor(255, 255, 255))
        p.drawText(QRectF(1400, 20, 500, 35), Qt.AlignRight,
                   f"Iteration: {self.mw.current_iter + 1}/{self.mw.total_iter}")
        p.setPen(QColor(255, 215, 0))
        p.drawText(QRectF(1400, 55, 500, 35), Qt.AlignRight,
                   f"Best Acc: {self.mw.best_acc:.1%}")

        # Widget pixmaps
        pix_cur = self.mw.viz_cur.grab()
        pix_best = self.mw.viz_best.grab()
        pix_net = self.mw.net_viz.grab()
        pix_scat = self.mw.history.grab()
        pix_hist = self.mw.histogram.grab()

        def _draw_scaled(painter, pixmap, target_rect):
            scaled = pixmap.scaled(
                int(target_rect.width()), int(target_rect.height()),
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            x = target_rect.x() + (target_rect.width() - scaled.width()) / 2
            y = target_rect.y() + (target_rect.height() - scaled.height()) / 2
            painter.drawPixmap(QPointF(x, y), scaled)

        _draw_scaled(p, pix_cur, QRectF(10, 110, 950, 480))
        _draw_scaled(p, pix_best, QRectF(960, 110, 950, 480))
        _draw_scaled(p, pix_net, QRectF(10, 600, 630, 480))
        _draw_scaled(p, pix_scat, QRectF(640, 600, 640, 480))
        _draw_scaled(p, pix_hist, QRectF(1280, 600, 640, 480))

        # Border rectangles
        p.setPen(QPen(QColor(60, 60, 80), 2))
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(10, 110, 950, 480))
        p.drawRect(QRectF(960, 110, 950, 480))
        p.drawRect(QRectF(10, 600, 630, 480))
        p.drawRect(QRectF(640, 600, 640, 480))
        p.drawRect(QRectF(1280, 600, 640, 480))

        p.end()
        return canvas

    # ── Frame Writing ─────────────────────────────────────────────────────────

    def _write_frame(self, canvas):
        """Convert a QImage to BGR numpy array and write to video."""
        if not self.video_writer:
            return
        canvas = canvas.convertToFormat(QImage.Format_RGB888)
        width = canvas.width()
        height = canvas.height()

        ptr = canvas.bits()
        arr = np.frombuffer(ptr, dtype=np.uint8)

        bpl = canvas.bytesPerLine()
        if bpl == width * 3:
            arr = arr.reshape(height, width, 3)
        else:
            arr = arr.reshape(height, bpl)
            arr = arr[:, :width * 3].reshape(height, width, 3)

        frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame)