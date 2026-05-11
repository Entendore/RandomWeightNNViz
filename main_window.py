"""
Main application window — orchestrates simulation, widgets, and export.
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QSpinBox, QGroupBox, QSplitter,
    QDoubleSpinBox, QGridLayout, QProgressBar, QCheckBox, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QPalette, QColor

from neural_network import NeuralNetwork
from data_generator import DataGenerator
from drawing_dialog import DrawingDialog
from network_viz import NetworkVizWidget
from data_viz import DataVizWidget
from history_widget import HistoryWidget
from histogram_widget import HistogramWidget
from video_exporter import VideoExporter, CV2_AVAILABLE


class MainWindow(QMainWindow):
    METHODS = ["he", "xavier", "lecun", "uniform", "normal_01", "normal_1", "large", "sparse"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Random Weight Search + Digits/Hands/Custom + Auto MP4")
        self.setMinimumSize(1100, 720)
        self.resize(1400, 900)

        # Simulation state
        self.nn = None
        self.best_nn = None
        self.X = self.y = None
        self.running = False
        self.current_iter = 0
        self.total_iter = 100
        self.best_acc = -1
        self.current_acc = 0
        self.current_loss = 0

        # Video exporter
        self.exporter = VideoExporter(self)

        self._build_ui()
        self._build_timers()
        self._full_reset()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self._apply_palette()
        self._apply_stylesheet()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ── LEFT PANEL ────────────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(255)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(4)

        # Dataset group
        g = QGroupBox("Dataset")
        gl = QVBoxLayout(g)
        gl.addWidget(QLabel("Type:"))
        self.dataset_cb = QComboBox()
        self.dataset_cb.addItems([
            "xor", "circle", "spiral", "moons", "gaussian",
            "digits_4x4", "hand_signs", "custom_draw",
        ])
        self.dataset_cb.currentTextChanged.connect(self._on_data_changed)
        gl.addWidget(self.dataset_cb)

        self.draw_btn = QPushButton("✏️ Open Drawing Canvas")
        self.draw_btn.setStyleSheet(
            "QPushButton{background:#2a6090;color:#fff;padding:5px;border-radius:3px}"
            "QPushButton:hover{background:#3a70a0}"
        )
        self.draw_btn.clicked.connect(self._open_draw_dialog)
        gl.addWidget(self.draw_btn)

        hl = QHBoxLayout()
        hl.addWidget(QLabel("Samples:"))
        self.nspin = QSpinBox()
        self.nspin.setRange(50, 2000)
        self.nspin.setValue(400)
        self.nspin.setSingleStep(50)
        self.nspin.valueChanged.connect(self._on_data_changed)
        hl.addWidget(self.nspin)
        gl.addLayout(hl)
        lv.addWidget(g)

        # Architecture group
        g = QGroupBox("Architecture")
        gl = QGridLayout(g)
        self.in_label = QLabel("Input: 2")
        gl.addWidget(self.in_label, 0, 0, 1, 2)
        for r, (label, default) in enumerate(
            [("Hidden 1:", 8), ("Hidden 2:", 8), ("Hidden 3:", 0)], 1
        ):
            gl.addWidget(QLabel(label), r, 0)
            sp = QSpinBox()
            sp.setRange(0, 64)
            sp.setValue(default)
            sp.valueChanged.connect(self._on_arch_changed)
            gl.addWidget(sp, r, 1)
            setattr(self, f"h{r}_spin", sp)
        self.out_label = QLabel("Output: 1")
        gl.addWidget(self.out_label, 4, 0, 1, 2)
        lv.addWidget(g)

        # Randomization group
        g = QGroupBox("Randomization")
        gl = QVBoxLayout(g)
        gl.addWidget(QLabel("Init method:"))
        self.method_cb = QComboBox()
        self.method_cb.addItems(self.METHODS)
        gl.addWidget(self.method_cb)
        self.compare_cb = QCheckBox("Cycle ALL methods")
        gl.addWidget(self.compare_cb)
        lv.addWidget(g)

        # Interactive Search group
        g = QGroupBox("Interactive Search")
        gl = QVBoxLayout(g)

        hl = QHBoxLayout()
        hl.addWidget(QLabel("Iterations:"))
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(10, 10000)
        self.iter_spin.setValue(200)
        self.iter_spin.setSingleStep(50)
        hl.addWidget(self.iter_spin)
        gl.addLayout(hl)

        hl = QHBoxLayout()
        hl.addWidget(QLabel("Speed:"))
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 200)
        self.speed_spin.setValue(3)
        self.speed_spin.setSuffix(" / tick")
        hl.addWidget(self.speed_spin)
        gl.addLayout(hl)

        self.start_btn = QPushButton("🎲  Start Search")
        self.start_btn.setStyleSheet(
            "QPushButton{background:#2a7090;color:#fff;padding:8px;border-radius:4px;"
            "font-weight:bold;font-size:12px}QPushButton:hover{background:#3a80a0}"
        )
        self.start_btn.clicked.connect(self._toggle_run)
        gl.addWidget(self.start_btn)

        self.step_btn = QPushButton("→  Single Step")
        self.step_btn.setStyleSheet(
            "QPushButton{background:#505068;color:#ddd;padding:5px;border-radius:3px}"
            "QPushButton:hover{background:#606080}"
        )
        self.step_btn.clicked.connect(self._single_step)
        gl.addWidget(self.step_btn)

        self.reset_btn = QPushButton("↺  Full Reset")
        self.reset_btn.setStyleSheet(
            "QPushButton{background:#804040;color:#fff;padding:5px;border-radius:3px}"
            "QPushButton:hover{background:#905050}"
        )
        self.reset_btn.clicked.connect(self._full_reset)
        gl.addWidget(self.reset_btn)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        gl.addWidget(self.progress)
        lv.addWidget(g)

        # Live Stats group
        g = QGroupBox("Live Stats")
        gl = QVBoxLayout(g)
        self.cur_label = QLabel("Current:  —")
        self.cur_label.setStyleSheet("color:#aaa;font-size:11px")
        gl.addWidget(self.cur_label)
        self.best_label = QLabel("Best:  —")
        self.best_label.setStyleSheet("color:#ffd700;font-size:12px;font-weight:bold")
        gl.addWidget(self.best_label)
        self.iter_label = QLabel("Iteration: 0")
        self.iter_label.setStyleSheet("color:#999;font-size:10px")
        gl.addWidget(self.iter_label)
        lv.addWidget(g)

        # Auto MP4 Export group
        g = QGroupBox("Auto MP4 Export (Full Run)")
        gl = QVBoxLayout(g)
        if not CV2_AVAILABLE:
            gl.addWidget(QLabel("⚠ pip install opencv-python\nto enable recording"))
            self.export_btn = QPushButton("Export (Missing CV2)")
            self.export_btn.setEnabled(False)
        else:
            gl.addWidget(QLabel("Automatically runs all iterations\nand saves 1080p MP4 to /output/"))
            self.export_btn = QPushButton("🎬  Export Full Run to MP4")
        self.export_btn.setStyleSheet(
            "QPushButton{background:#a02020;color:#fff;padding:8px;border-radius:4px;"
            "font-weight:bold;font-size:12px}QPushButton:hover{background:#c03030}"
        )
        self.export_btn.clicked.connect(self._start_auto_export)
        gl.addWidget(self.export_btn)
        self.export_status = QLabel("")
        self.export_status.setStyleSheet("color:#ff8888;font-size:10px")
        gl.addWidget(self.export_status)
        lv.addWidget(g)

        lv.addStretch()
        root.addWidget(left)

        # ── RIGHT PANEL ───────────────────────────────────────────────────
        splitter = QSplitter(Qt.Vertical)

        row1 = QSplitter(Qt.Horizontal)
        self.viz_cur = DataVizWidget()
        self.viz_best = DataVizWidget()
        row1.addWidget(self.viz_cur)
        row1.addWidget(self.viz_best)
        row1.setSizes([500, 500])

        row2 = QSplitter(Qt.Horizontal)
        self.history = HistoryWidget()
        self.histogram = HistogramWidget()
        row2.addWidget(self.history)
        row2.addWidget(self.histogram)
        row2.setSizes([500, 300])

        self.net_viz = NetworkVizWidget()

        splitter.addWidget(row1)
        splitter.addWidget(row2)
        splitter.addWidget(self.net_viz)
        splitter.setSizes([320, 220, 280])
        root.addWidget(splitter, 1)

        self.statusBar().setStyleSheet("color:#aaa;")

    def _apply_palette(self):
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(34, 34, 46))
        pal.setColor(QPalette.WindowText, QColor(200, 200, 220))
        pal.setColor(QPalette.Base, QColor(30, 30, 42))
        pal.setColor(QPalette.Text, QColor(200, 200, 220))
        pal.setColor(QPalette.Button, QColor(48, 48, 62))
        pal.setColor(QPalette.ButtonText, QColor(200, 200, 220))
        self.setPalette(pal)

    def _apply_stylesheet(self):
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
            QDialog { background-color: #2a2a3a; color: #ddd; }
            QRadioButton { color: #ccc; }
        """)

    def _build_timers(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(25)

        self.export_timer = QTimer()
        self.export_timer.timeout.connect(self._export_tick)
        self.export_timer.setInterval(5)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _layer_sizes(self):
        ds = self.dataset_cb.currentText()
        n_in, n_out = (
            DataGenerator.get_shape(ds)
            if ds != "custom_draw"
            else (
                self.X.shape[1] if self.X is not None else 2,
                self.y.shape[1] if self.y is not None else 1,
            )
        )
        layers = [n_in]
        for sp in (self.h1_spin, self.h2_spin, self.h3_spin):
            if sp.value() > 0:
                layers.append(sp.value())
        layers.append(n_out)
        return layers

    def _get_method(self):
        if self.compare_cb.isChecked():
            return self.METHODS[self.current_iter % len(self.METHODS)]
        return self.method_cb.currentText()

    def _update_arch_labels(self):
        layers = self._layer_sizes()
        self.in_label.setText(f"Input: {layers[0]}")
        self.out_label.setText(f"Output: {layers[-1]}")

    # ── State Management ──────────────────────────────────────────────────────

    def _full_reset(self):
        self._stop()
        self._stop_export()

        self.current_iter = 0
        self.best_acc = -1
        self.best_nn = None

        ds = self.dataset_cb.currentText()
        if ds == "custom_draw" and (self.X is None or len(self.X) == 0):
            self.X, self.y = DataGenerator.generate("gaussian", self.nspin.value())
        elif ds != "custom_draw":
            self.X, self.y = DataGenerator.generate(ds, self.nspin.value())

        self._update_arch_labels()

        self.nn = NeuralNetwork(self._layer_sizes())
        self.nn.randomize(self.method_cb.currentText())
        acc, loss = self.nn.evaluate(self.X, self.y)
        self.current_acc = acc
        self.current_loss = loss

        self.viz_cur.set_data(self.X, self.y)
        self.viz_best.set_data(self.X, self.y)
        self.viz_cur.update_network(self.nn, "Current Candidate", acc)
        self.viz_best.update_network(None, "Best Found (none yet)")
        self.net_viz.set_network(self.nn, "Current Candidate")

        self.history.clear()
        self.histogram.clear()

        self.cur_label.setText(f"Current:  Acc {acc:.1%}  Loss {loss:.4f}")
        self.best_label.setText("Best:  —")
        self.iter_label.setText("Iteration: 0")

    def _refresh_current(self, method):
        acc, loss = self.nn.evaluate(self.X, self.y)
        self.current_acc = acc
        self.current_loss = loss

        is_new_best = acc > self.best_acc
        if is_new_best:
            self.best_acc = acc
            if self.best_nn is None:
                self.best_nn = NeuralNetwork(self._layer_sizes())
            self.best_nn.copy_from(self.nn)
            self.viz_best.update_network(self.best_nn, "★ Best Found", acc)

        self.viz_cur.update_network(self.nn, f"#{self.current_iter + 1} ({method})", acc)
        self.net_viz.set_network(
            self.nn, f"Candidate #{self.current_iter + 1}  [{method}]",
            is_best=is_new_best,
        )
        self.history.add(self.current_iter, method, acc, loss)
        self.histogram.add(acc)

        self.cur_label.setText(f"Current:  Acc {acc:.1%}  Loss {loss:.4f}")
        if self.best_nn:
            self.best_label.setText(
                f"Best:  Acc {self.best_acc:.1%}  "
                f"(iter #{self.history.attempts[self.history.best_idx][0] + 1})"
            )
        self.iter_label.setText(f"Iteration: {self.current_iter + 1} / {self.total_iter}")

        if is_new_best:
            self.statusBar().showMessage(
                f"🎉 NEW BEST at #{self.current_iter + 1}!  Acc = {acc:.1%}  ({method})"
            )
        else:
            self.statusBar().showMessage(
                f"#{self.current_iter + 1}  Acc={acc:.1%}  "
                f"Best={self.best_acc:.1%}  ({method})"
            )

    # ── Interactive Simulation ────────────────────────────────────────────────

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
            "QPushButton{background:#c09020;color:#fff;padding:8px;border-radius:4px;"
            "font-weight:bold;font-size:12px}QPushButton:hover{background:#d0a030}"
        )
        self.timer.start()

    def _stop(self):
        self.timer.stop()
        self.running = False
        self.start_btn.setText("🎲  Start Search")
        self.start_btn.setStyleSheet(
            "QPushButton{background:#2a7090;color:#fff;padding:8px;border-radius:4px;"
            "font-weight:bold;font-size:12px}QPushButton:hover{background:#3a80a0}"
        )
        if self.current_iter >= self.total_iter:
            self.progress.setVisible(False)
            self.statusBar().showMessage(
                f"Search complete!  Best accuracy: {self.best_acc:.1%}"
            )

    def _tick(self):
        if not self.running:
            return
        for _ in range(self.speed_spin.value()):
            if self.current_iter >= self.total_iter:
                self._stop()
                return
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

    # ── Drawing Dialog ────────────────────────────────────────────────────────

    def _open_draw_dialog(self):
        dlg = DrawingDialog(self)
        if dlg.exec() == DrawingDialog.Accepted:
            X, y = dlg.get_data()
            if X is not None:
                self.dataset_cb.setCurrentText("custom_draw")
                self.X, self.y = X, y
                self._full_reset()

    # ── Automated Video Export ────────────────────────────────────────────────

    def _start_auto_export(self):
        if not CV2_AVAILABLE or self.exporter.is_exporting:
            return

        ds = self.dataset_cb.currentText()

        # Reset simulation for the export
        self.current_iter = 0
        self.best_acc = -1
        self.best_nn = None
        self.X, self.y = DataGenerator.generate(ds, self.nspin.value())
        self._update_arch_labels()
        self.nn = NeuralNetwork(self._layer_sizes())
        self.nn.randomize(self.method_cb.currentText())
        self.viz_cur.set_data(self.X, self.y)
        self.viz_best.set_data(self.X, self.y)
        self.history.clear()
        self.histogram.clear()
        self.total_iter = self.iter_spin.value()
        self.progress.setVisible(True)
        self.progress.setRange(0, self.total_iter)
        self.progress.setValue(0)

        # Start exporter (writes intro frames)
        success = self.exporter.start(ds, self.total_iter)
        if not success:
            self.statusBar().showMessage("Error: Could not open video writer.")
            return

        self.export_btn.setText("⏹  Exporting...")
        self.export_btn.setEnabled(False)
        self.export_status.setText(f"Writing to {self.exporter.video_filename}")
        self.export_timer.start()

    def _export_tick(self):
        if not self.exporter.is_exporting:
            return

        # One simulation step
        method = self._get_method()
        self.nn = NeuralNetwork(self._layer_sizes())
        self.nn.randomize(method)
        self._refresh_current(method)

        # Write frames
        is_new_best = self.current_acc >= self.best_acc
        self.exporter.write_iteration_frames(is_new_best=is_new_best)

        self.current_iter += 1
        self.progress.setValue(self.current_iter)

        if self.current_iter >= self.total_iter:
            self._stop_export()

    def _stop_export(self):
        if not self.exporter.is_exporting:
            return
        self.export_timer.stop()

        filename = self.exporter.finish()

        self.export_btn.setText("🎬  Export Full Run to MP4")
        self.export_btn.setEnabled(True)
        self.export_status.setText(f"✅ Saved: {filename}")
        self.progress.setVisible(False)
        self.statusBar().showMessage(f"MP4 Export Complete: {filename}")