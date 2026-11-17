"""阶段四桌面入口。采集和推理在工作线程中，界面只接收本地内存状态。"""

import json
import os
from pathlib import Path
from queue import Empty, Queue
import sqlite3
import sys
import time
import ctypes
from ctypes import wintypes

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QAbstractNativeEventFilter
from PySide6.QtGui import QIcon, QImage, QPixmap, QCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTabWidget, QFormLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QSystemTrayIcon, QMenu, QFrame,
)

from app import ROOT, arguments, create_landmarker, open_camera, frame_problem
from calibration_store import CalibrationStore
from mouth import MouthMonitor
from posture import Config, DistanceExperiment, Reading, measure_face
from reminders import ReminderController, AccumulatedReminder

DATA = Path(os.environ.get("LOCALAPPDATA", str(ROOT))) / "SitSense"
DEFAULTS = dict(camera=0, backend="dshow", threshold=1.15, alert_after=30,
                alert_interval=300, mouth_after=10, mouth_interval=300,
                posture=True, mouth=True, preview=True)


class Worker(QThread):
    snapshot = Signal(dict)
    alert = Signal(str)
    failed = Signal(str)

    def __init__(self, settings):
        super().__init__()
        self.settings = dict(settings)
        self.commands = Queue()

    def run(self):
        import cv2
        import mediapipe as mp

        capture = None
        try:
            cv2.setNumThreads(1)
            config = Config(near_ratio=self.settings["threshold"])
            args, _ = arguments([])
            args.camera = self.settings["camera"]
            args.backend = self.settings["backend"]
            distance = DistanceExperiment(config)
            mouth = MouthMonitor()
            calibration_store = None
            last_saved_posture = None
            last_saved_mouth = None
            calibration_save_error = ""
            sitting = ReminderController(self.settings["alert_after"], self.settings["alert_interval"], self.settings["posture"])
            oral = AccumulatedReminder(self.settings["mouth_after"], self.settings["mouth_interval"], self.settings["mouth"])
            paused = False
            preview = False
            next_sample = 0.0
            last_timestamp = -1
            seen = {"坐直": None, "撕嘴皮": None}
            with create_landmarker(args.model, config, mouth_enabled=self.settings["mouth"]) as detector:
                if self.isInterruptionRequested():
                    return
                capture = open_camera(args, config)
                actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or config.camera_width
                actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config.camera_height
                calibration_store = CalibrationStore(
                    DATA / "calibration.json",
                    camera=args.camera,
                    backend=args.backend,
                    frame_width=actual_width,
                    frame_height=actual_height,
                )
                if calibration_store.posture_baseline is not None:
                    distance.load_baseline(calibration_store.posture_baseline)
                if calibration_store.mouth_baseline is not None:
                    mouth.load_baseline(calibration_store.mouth_baseline)
                last_saved_posture = distance.baseline
                last_saved_mouth = mouth.baseline
                while not self.isInterruptionRequested():
                    while True:
                        try:
                            command, value = self.commands.get_nowait()
                        except Empty:
                            break
                        if command == "preview":
                            preview = value
                        elif command == "pause":
                            paused = value
                            sitting.set_paused(paused)
                            oral.set_paused(paused)
                            mouth.unknown("已暂停" if paused else "等待画面")
                            distance.invalidate("等待画面")
                            if capture is not None:
                                capture.release()
                                capture = None
                            if not paused:
                                capture = open_camera(args, config)
                            self.snapshot.emit(dict(paused=paused, camera="已释放" if paused else "正在使用"))
                        elif command == "calibrate" and not paused:
                            distance.start_calibration(time.monotonic())
                            # 即使新校准结果与旧数值相同，也允许重新写入校准时间和文件。
                            last_saved_posture = None
                            sitting.reset_continuity()
                            oral.reset_continuity()
                            if self.settings["mouth"]:
                                mouth.calibrate(time.monotonic())
                                last_saved_mouth = None
                    if paused:
                        self.msleep(50)
                        continue
                    grabbed = capture.grab()
                    now = time.monotonic()
                    if now < next_sample:
                        self.msleep(10)
                        continue
                    next_sample = now + 1 / config.detection_fps
                    ok, frame = capture.retrieve() if grabbed else (False, None)
                    problem = frame_problem(frame if ok else None, config)
                    mouth_state = "已关闭"
                    bounds = None
                    mouth_bounds = None
                    if problem:
                        reading = distance.update(now, None, problem)
                        oral.reset_continuity()
                        if self.settings["mouth"]:
                            mouth_state = mouth.skip("画面不可用", now)
                    else:
                        timestamp = max(last_timestamp + 1, int(now * 1000))
                        last_timestamp = timestamp
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
                        if time.monotonic() - now > config.stale_seconds:
                            measurement, reason = None, "推理过慢，等待新画面"
                        else:
                            measurement, bounds, reason = measure_face(result.face_landmarks, frame.shape[1], frame.shape[0], config)
                        reading = distance.update(now, measurement, reason)
                        if self.settings["mouth"]:
                            mouth_state = mouth.update(result, now) if measurement is not None else mouth.skip(reason, now)
                        if measurement is not None and mouth_state == "疑似异常":
                            lips = [result.face_landmarks[0][i] for i in (61, 291, 0, 17, 13, 14)]
                            h, w = frame.shape[:2]
                            mouth_bounds = (int(min(p.x for p in lips)*w)-8, int(min(p.y for p in lips)*h)-8,
                                            int(max(p.x for p in lips)*w)+8, int(max(p.y for p in lips)*h)+8)
                        del result, rgb
                    # 只有完整校准成功后才保存数值基准；失败或中断不会覆盖上次文件。
                    if distance.baseline is not None and distance.baseline != last_saved_posture:
                        last_saved_posture = distance.baseline
                        try:
                            calibration_store.save_posture(distance.baseline)
                            calibration_save_error = ""
                        except (OSError, ValueError):
                            calibration_save_error = "坐姿校准已完成，但本地保存失败"
                    if mouth.baseline is not None and mouth.baseline != last_saved_mouth:
                        last_saved_mouth = mouth.baseline
                        try:
                            calibration_store.save_mouth(mouth.baseline)
                            calibration_save_error = ""
                        except (OSError, ValueError):
                            calibration_save_error = "口部校准已完成，但本地保存失败"

                    sitting.update(reading.state, now)
                    oral.update("疑似过近" if mouth_state == "疑似异常" else mouth_state, now)
                    for label, controller in (("坐直", sitting), ("撕嘴皮", oral)):
                        if controller.last_notice_at != seen[label]:
                            seen[label] = controller.last_notice_at
                            self.alert.emit(label)
                    image = None
                    if preview and not problem:
                        rgb_preview = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                        height, width = rgb_preview.shape[:2]
                        if bounds is not None:
                            left, top, right, bottom = bounds
                            color = (255, 165, 55) if reading.state == "疑似过近" else ((70, 210, 145) if reading.state == "正常" else (160, 170, 185))
                            cv2.rectangle(rgb_preview, (width-1-right, top), (width-1-left, bottom), color, 2)
                        if mouth_bounds is not None:
                            left, top, right, bottom = mouth_bounds
                            cv2.rectangle(rgb_preview, (width-1-right, top), (width-1-left, bottom), (255, 165, 55), 3)
                        image = QImage(rgb_preview.data, width, height, rgb_preview.strides[0], QImage.Format.Format_RGB888).copy()
                    self.snapshot.emit(dict(
                        paused=False, camera="画面不可用" if problem else "正在使用",
                        posture=reading.state, reason=reading.reason, mouth=mouth_state,
                        mouth_reason=mouth.reason if self.settings["mouth"] else "已关闭",
                        ratio=reading.smooth_ratio, elapsed=sitting.view(now).elapsed,
                        mouth_elapsed=oral.view(now).elapsed, image=image, at=now,
                        posture_calibrated=distance.baseline is not None,
                        mouth_calibrated=mouth.baseline is not None,
                        calibration_status=calibration_store.status,
                        calibration_error=calibration_save_error,
                    ))
        except Exception as error:
            # 不打印第三方异常载荷，防止特征数据进入日志。
            self.failed.emit(f"监测已停止（{type(error).__name__}）。请检查相机占用、模型或设置，再点恢复。")
        finally:
            if capture is not None:
                capture.release()


class MSG(ctypes.Structure):
    _fields_ = [("hwnd", wintypes.HWND), ("message", wintypes.UINT),
                ("wParam", wintypes.WPARAM), ("lParam", wintypes.LPARAM),
                ("time", wintypes.DWORD), ("pt", wintypes.POINT)]


class EndSessionFilter(QAbstractNativeEventFilter):
    """Windows 注销/卸载请求结束会话时真正退出，而不是隐藏到托盘占用文件。"""

    def __init__(self, window):
        super().__init__()
        self.window = window

    def nativeEventFilter(self, eventType, message):
        if eventType == b"windows_generic_MSG":
            msg = MSG.from_address(int(message))
            if msg.message in (0x0011, 0x0016):
                self.window.exit_app()
        return False, 0


class Notice(QFrame):
    def __init__(self, label, asset, row):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint |
                         Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedSize(184, 68)
        self.setStyleSheet("QFrame { background: #fff6e6; border: 1px solid #e4cda4; border-radius: 6px; } QLabel { border: 0; font: 20px 'Microsoft YaHei'; color: #334155; }")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(12)
        icon = QLabel()
        icon.setPixmap(QIcon(str(ROOT / "assets" / asset)).pixmap(32, 32))
        layout.addWidget(icon)
        layout.addWidget(QLabel(label))
        self.row = row
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide)

    def present(self):
        screen = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
        area = screen.availableGeometry()
        self.move(area.center().x() - self.width() // 2, area.center().y() - self.height() // 2 + self.row * 76)
        self.show()
        self.timer.start(8000)


class Window(QMainWindow):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.worker = None
        self.paused = False
        self.exiting = False
        self.restarting = False
        self.last_sample = None
        self.previous_states = None
        self.totals = {"正常": 0.0, "异常": 0.0, "无法判断": 0.0}
        self.count = 0
        self.setWindowTitle("SitSense")
        self.resize(1120, 680)
        self.setWindowIcon(QIcon(str(ROOT / "assets" / "shrimp.svg")))
        self.setStyleSheet("QWidget { font: 14px 'Microsoft YaHei'; } QPushButton { padding: 7px 14px; } QTabWidget::pane { border: 0; } QSpinBox, QDoubleSpinBox, QComboBox { padding: 5px; }")
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        home = QWidget()
        split = QHBoxLayout(home)
        split.setContentsMargins(12, 12, 12, 12)
        panel = QWidget()
        panel.setFixedWidth(340)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(12)
        self.status = QLabel("正在加载本地模型…")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.detail = QLabel("摄像头准备中")
        self.detail.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addWidget(self.detail)
        buttons = QHBoxLayout()
        self.pause_button = QPushButton("暂停")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.calibrate_button = QPushButton("校准")
        self.calibrate_button.clicked.connect(self.calibrate)
        buttons.addWidget(self.pause_button)
        buttons.addWidget(self.calibrate_button)
        layout.addLayout(buttons)
        self.preview = QLabel("预览已关闭；仍在本地监测")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(480, 360)
        from PySide6.QtWidgets import QSizePolicy
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.preview.setStyleSheet("background: #141b27; color: #aab8cc; border-radius: 10px;")
        split.addWidget(self.preview, 1)
        split.addWidget(panel)
        self.statistics = QLabel()
        self.statistics.setWordWrap(True)
        layout.addWidget(self.statistics)
        hint = QLabel("首次校准后会在本机保存数值基准，下次启动自动加载。\n更换摄像头或接口后请重新校准。\n全程离线 · 不保存画面与人脸数据")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #64748b; font-size: 12px;")
        layout.addWidget(hint)
        self.tabs.addTab(home, "监测")
        self.make_settings()
        self.notices = {"坐直": Notice("坐直", "shrimp.svg", 0), "撕嘴皮": Notice("撕嘴皮", "mouth-off.svg", 1)}
        self.tray = QSystemTrayIcon(self.windowIcon(), self)
        self.tray.setToolTip("SitSense · 静音监测")
        menu = QMenu(self)
        menu.addAction("打开主界面", self.restore)
        self.pause_action = menu.addAction("暂停", self.toggle_pause)
        menu.addAction("校准", self.calibrate)
        menu.addSeparator()
        menu.addAction("退出", self.exit_app)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(lambda reason: self.restore() if reason == QSystemTrayIcon.ActivationReason.DoubleClick else None)
        self.tray.show()
        DATA.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(DATA / "statistics.sqlite3")
        self.db.execute("CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY, started TEXT, normal REAL, abnormal REAL, unknown REAL, reminders INTEGER)")
        self.db.execute("CREATE TABLE IF NOT EXISTS events (occurred TEXT, kind TEXT)")
        self.session_id = self.db.execute("INSERT INTO sessions VALUES (NULL, datetime('now','localtime'),0,0,0,0)").lastrowid
        self.db.commit()
        historic = self.db.execute("SELECT COALESCE(SUM(normal),0), COALESCE(SUM(abnormal),0), COALESCE(SUM(unknown),0), COALESCE(SUM(reminders),0) FROM sessions WHERE id<>?", (self.session_id,)).fetchone()
        history = QLabel(f"历史累计：正常 {historic[0]/60:.1f} 分 · 异常 {historic[1]/60:.1f} 分\n无法判断 {historic[2]/60:.1f} 分 · 提醒 {historic[3]} 次")
        history.setWordWrap(True)
        layout.addWidget(history)
        self.save_timer = QTimer(self)
        self.save_timer.timeout.connect(self.save_statistics)
        self.save_timer.start(30000)
        self.start_worker()

    def make_settings(self):
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(22, 18, 22, 18)
        self.controls = {}
        fields = (("camera", "摄像头索引", 0, 10), ("threshold", "距离放大阈值", 1.02, 2.0),
                  ("alert_after", "坐直提醒：持续秒数", 1, 600), ("alert_interval", "坐直提醒：间隔秒数", 8, 3600),
                  ("mouth_after", "口部提醒：秒数（含短暂跳变）", 2, 120), ("mouth_interval", "口部提醒：间隔秒数", 8, 3600))
        for key, label, low, high in fields:
            control = QDoubleSpinBox() if key == "threshold" else QSpinBox()
            control.setRange(low, high)
            if key == "threshold":
                control.setSingleStep(0.01)
            control.setValue(self.settings[key])
            self.controls[key] = control
            form.addRow(label, control)
        backend = QComboBox()
        backend.addItems(["dshow", "msmf", "any"])
        backend.setCurrentText(self.settings["backend"])
        self.controls["backend"] = backend
        form.addRow("相机接口（黑屏时可切换）", backend)
        for key, label in (("posture", "坐直提醒"), ("mouth", "撕嘴皮检测（简化版）"), ("preview", "显示本地摄像头预览")):
            control = QCheckBox(label)
            control.setChecked(self.settings[key])
            self.controls[key] = control
            form.addRow(control)
        self.apply_button = QPushButton("保存并应用")
        self.apply_button.clicked.connect(self.apply_settings)
        form.addRow(self.apply_button)
        note = QLabel("应用后会重启监测；摄像头配置不变时自动沿用已保存的校准基准。\n更换摄像头或接口后需要重新校准。")
        note.setWordWrap(True)
        form.addRow(note)
        self.tabs.addTab(page, "设置")

    def start_worker(self):
        self.last_sample = None
        self.previous_states = None
        self.paused = False
        self.pause_button.setText("暂停")
        self.pause_action.setText("暂停")
        self.calibrate_button.setEnabled(True)
        self.worker = Worker(self.settings)
        self.worker.snapshot.connect(self.receive)
        self.worker.alert.connect(self.remind)
        self.worker.failed.connect(self.failure)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()
        self.sync_preview()

    def sync_preview(self):
        if self.worker:
            self.worker.commands.put(("preview", self.settings["preview"] and self.isVisible() and not self.isMinimized()))

    def receive(self, data):
        if self.exiting or self.restarting:
            return
        if "posture" not in data:
            self.status.setText("已暂停" if data["paused"] else "正在恢复监测")
            self.detail.setText("摄像头：" + data["camera"])
            self.preview.clear()
            self.last_sample = None
            return
        posture = data["posture"]
        oral = data["mouth"]
        state = "异常" if posture == "疑似过近" or oral == "疑似异常" else (
            "无法判断" if posture == "无法判断" or (self.settings["mouth"] and oral == "无法判断") else "正常")
        now = data["at"]
        if self.last_sample is not None and 0 < now - self.last_sample <= 0.8:
            self.totals[self.previous_states] += now - self.last_sample
        self.last_sample, self.previous_states = now, state
        self.status.setText(f"坐姿：{posture}　口部：{oral}")
        ratio = "--" if data["ratio"] is None else f"{data['ratio']:.2f}"
        posture_calibration = "已保存" if data.get("posture_calibrated") else "未校准"
        mouth_calibration = "已保存" if data.get("mouth_calibrated") else "未校准"
        calibration_line = f"本地校准：坐姿{posture_calibration} · 口部{mouth_calibration}"
        calibration_status = data.get("calibration_status", "")
        if calibration_status and not data.get("posture_calibrated") and not data.get("mouth_calibrated"):
            calibration_line += f"（{calibration_status}）"
        save_error = data.get("calibration_error", "")
        if save_error:
            calibration_line += f" · {save_error}"
        self.detail.setText(
            f"摄像头：{data['camera']}　距离倍数：{ratio}\n"
            f"{data['reason']}\n{calibration_line}\n"
            f"口部：{data['mouth_reason']}\n"
            f"持续过近 {data['elapsed']:.0f} 秒　口部提醒计时 {data['mouth_elapsed']:.0f} 秒"
        )
        if data["image"] is not None and self.isVisible() and self.settings["preview"]:
            self.preview.setPixmap(QPixmap.fromImage(data["image"]).scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))
        else:
            self.preview.setText("预览已关闭；仍在本地监测" if not self.settings["preview"] else "等待有效画面")
        self.statistics.setText(f"本次统计：正常 {self.totals['正常']/60:.1f} 分钟 · 异常 {self.totals['异常']/60:.1f} 分钟\n无法判断 {self.totals['无法判断']/60:.1f} 分钟 · 提醒 {self.count} 次")
        if posture != "疑似过近":
            self.notices["坐直"].hide()
        if oral == "已关闭":
            self.notices["撕嘴皮"].hide()

    def remind(self, label):
        if self.exiting or self.restarting or self.paused:
            return
        self.notices[label].present()
        self.count += 1
        self.db.execute("INSERT INTO events VALUES (datetime('now','localtime'),?)", (label,))
        self.save_statistics()

    def save_statistics(self):
        self.db.execute("UPDATE sessions SET normal=?, abnormal=?, unknown=?, reminders=? WHERE id=?",
                        (self.totals["正常"], self.totals["异常"], self.totals["无法判断"], self.count, self.session_id))
        self.db.commit()

    def failure(self, message):
        self.status.setText(message)
        self.detail.setText("摄像头已释放")
        self.preview.clear()
        self.pause_button.setText("恢复")
        self.pause_action.setText("恢复")
        self.last_sample = None
        for notice in self.notices.values():
            notice.hide()

    def toggle_pause(self):
        if self.exiting or self.restarting:
            return
        if not self.worker.isRunning():
            self.start_worker()
            return
        self.paused = not self.paused
        self.worker.commands.put(("pause", self.paused))
        label = "恢复" if self.paused else "暂停"
        self.pause_button.setText(label)
        self.pause_action.setText(label)
        self.calibrate_button.setEnabled(not self.paused)
        for notice in self.notices.values():
            notice.hide()

    def calibrate(self):
        if self.worker and self.worker.isRunning() and not self.paused:
            self.worker.commands.put(("calibrate", None))
            for notice in self.notices.values():
                notice.hide()

    def apply_settings(self):
        for key, control in self.controls.items():
            self.settings[key] = control.isChecked() if isinstance(control, QCheckBox) else (
                control.currentText() if isinstance(control, QComboBox) else control.value())
        (DATA / "settings.json").write_text(json.dumps(self.settings, ensure_ascii=False, indent=2), encoding="utf-8")
        self.restarting = True
        self.apply_button.setEnabled(False)
        self.status.setText("正在应用设置…")
        for notice in self.notices.values():
            notice.hide()
        if self.worker.isRunning():
            self.worker.requestInterruption()
        else:
            self.worker_finished()

    def worker_finished(self):
        if self.exiting:
            self.save_statistics()
            self.db.close()
            QApplication.quit()
        elif self.restarting:
            self.restarting = False
            self.apply_button.setEnabled(True)
            self.calibrate_button.setEnabled(True)
            self.start_worker()

    def restore(self):
        self.showNormal()
        self.activateWindow()
        self.sync_preview()

    def closeEvent(self, event):
        if QSystemTrayIcon.isSystemTrayAvailable():
            event.ignore()
            self.hide()
            self.preview.clear()
            self.sync_preview()
        else:
            event.ignore()
            self.exit_app()

    def changeEvent(self, event):
        from PySide6.QtCore import QEvent
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange and self.isMinimized() and QSystemTrayIcon.isSystemTrayAvailable():
            self.hide()
            self.sync_preview()

    def exit_app(self):
        if self.exiting:
            return
        self.exiting = True
        self.save_statistics()
        self.save_timer.stop()
        self.tray.hide()
        for notice in self.notices.values():
            notice.hide()
        self.status.setText("正在退出并释放摄像头…")
        if self.worker.isRunning():
            self.worker.requestInterruption()
        else:
            self.worker_finished()


def main():
    cache_root = DATA if getattr(sys, "frozen", False) else ROOT
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / ".cache" / "matplotlib"))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if "--check" in sys.argv[1:]:
        import app
        return app.main(["--check"])
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    settings = dict(DEFAULTS)
    try:
        saved = json.loads((DATA / "settings.json").read_text(encoding="utf-8"))
        for key, default in DEFAULTS.items():
            value = saved.get(key, default)
            if type(value) is type(default):
                settings[key] = value
    except (OSError, ValueError, AttributeError):
        pass
    if not (DATA / "large-preview-v1").exists():
        settings["preview"] = True
        DATA.mkdir(parents=True, exist_ok=True)
        (DATA / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
        (DATA / "large-preview-v1").touch()
    if not (DATA / "mouth-grace-v1").exists():
        settings["mouth_after"] = 10
        DATA.mkdir(parents=True, exist_ok=True)
        (DATA / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
        (DATA / "mouth-grace-v1").touch()
    # 保留备用 MSMF 启动脚本和既有命令行参数。
    args, _ = arguments()
    for flag, key, value in (("--backend", "backend", args.backend), ("--camera", "camera", args.camera),
                             ("--threshold", "threshold", args.threshold), ("--alert-after", "alert_after", int(args.alert_after)),
                             ("--alert-interval", "alert_interval", int(args.alert_interval))):
        if any(item == flag or item.startswith(flag + "=") for item in sys.argv[1:]):
            settings[key] = value
    try:
        window = Window(settings)
    except Exception as error:
        # 无控制台模式下启动失败必须有可见提示；只报告异常类型，不输出第三方载荷。
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(None, "SitSense", f"启动失败（{type(error).__name__}）。请检查摄像头权限、本地模型和设置后重试。")
        return 1
    window.show()
    window.sync_preview()
    # 卸载或系统注销请求结束会话时必须真正退出，不能只隐藏到托盘占用文件。
    session_filter = EndSessionFilter(window)
    app.installNativeEventFilter(session_filter)
    app.commitDataRequest.connect(lambda session: window.exit_app())
    app.saveStateRequest.connect(lambda session: window.exit_app())
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
