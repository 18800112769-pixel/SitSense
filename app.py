"""SitSense：本地人脸距离检测和窗口内静音提醒。"""

import argparse
from collections import deque
from functools import lru_cache
import math
import os
from pathlib import Path
import sys
import time

from posture import Config, DistanceExperiment, Reading, measure_face
from reminders import ReminderController, ReminderView
from window_notice import SilentWindowNotice


# 打包后资源在 PyInstaller 的展开目录；源码运行时在项目根目录。
ROOT = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
WINDOW = "SitSense - Silent reminders (local only)"
PANEL_HEIGHT = 620


class UserError(Exception):
    """仅包含预先编写的提示，不包含第三方异常或监测数据。"""


def arguments(argv=None):
    parser = argparse.ArgumentParser(description="SitSense 离线距离检测与静音提醒")
    parser.add_argument("--camera", type=int, default=0, help="本地摄像头索引，默认 0")
    parser.add_argument("--backend", choices=("dshow", "msmf", "any"), default="dshow")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "face_landmarker.task")
    parser.add_argument("--threshold", type=float, default=Config.near_ratio, help="实验倍数阈值")
    parser.add_argument("--fps", type=float, default=Config.detection_fps, help="识别上限，5～10 次/秒")
    parser.add_argument("--calibration-seconds", type=float, default=Config.calibration_seconds)
    parser.add_argument("--smoothing-seconds", type=float, default=Config.smoothing_seconds)
    parser.add_argument("--alert-after", type=float, default=30.0, help="持续过近多少秒后提示，默认 30")
    parser.add_argument("--alert-interval", type=float, default=300.0, help="重复提示最小间隔秒数，默认 300")
    parser.add_argument("--no-reminders", action="store_true", help="启动时关闭提醒，窗口内按 R 切换")
    parser.add_argument("--check", action="store_true", help="仅检查本地模型和合成空白帧，不打开摄像头")
    args = parser.parse_args(argv)
    if args.camera < 0:
        parser.error("摄像头索引不能为负数；不支持网络摄像头地址。")
    if not all(math.isfinite(value) and value > 0 for value in (args.alert_after, args.alert_interval)):
        parser.error("持续提醒时间和重复间隔必须是有限的正数。")
    try:
        config = Config(
            near_ratio=args.threshold,
            detection_fps=args.fps,
            calibration_seconds=args.calibration_seconds,
            smoothing_seconds=args.smoothing_seconds,
        )
    except ValueError as error:
        parser.error(str(error))
    return args, config


def create_landmarker(model: Path, config: Config, mouth_enabled=False):
    import mediapipe as mp

    # 模型只能来自本地文件。缺失即报错，没有下载、缓存回退或在线推理路径。
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_buffer=model.read_bytes(),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=2,
        min_face_detection_confidence=config.detection_confidence,
        min_face_presence_confidence=config.presence_confidence,
        min_tracking_confidence=config.tracking_confidence,
        output_face_blendshapes=mouth_enabled,
        output_facial_transformation_matrixes=False,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def load_fonts():
    from PIL import ImageFont

    fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for filename in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        path = fonts / filename
        if path.is_file():
            return ImageFont.truetype(str(path), 20), ImageFont.truetype(str(path), 28)
    raise UserError("找不到本地中文字体，请检查 Windows 字体（微软雅黑或黑体）。")


def is_color_frame(frame) -> bool:
    import numpy as np

    return (
        frame is not None
        and frame.size > 0
        and frame.ndim == 3
        and frame.shape[2] == 3
        and frame.dtype == np.uint8
    )


def frame_problem(frame, config: Config) -> str:
    """只在本地检查取帧质量，不输出、保存像素或任何人脸信息。"""
    import numpy as np

    if frame is None or frame.size == 0:
        return "摄像头画面丢失，请检查连接或占用"
    if not is_color_frame(frame):
        return "摄像头画面格式异常，请尝试备用启动方式"
    # 稀疏采样只识别接近全黑的画面；这不是对具体硬件故障原因的诊断。
    if np.percentile(frame[::8, ::8], config.dark_percentile) < config.dark_pixel_threshold:
        return "画面过暗或接近全黑；请检查遮挡或尝试备用启动"
    return ""


@lru_cache(maxsize=4)
def panel_background(fonts, threshold: float):
    """只缓存固定文字和背景，不缓存摄像头画面、关键点或监测数值。"""
    from PIL import Image, ImageDraw

    panel = Image.new("RGB", (500, PANEL_HEIGHT), (20, 27, 39))
    draw = ImageDraw.Draw(panel)
    regular, large = fonts
    labels = (
        (16, "坐姿距离监测 · 静音提醒", large),
        (182, "平滑放大倍数", large),
        (230, "原始放大倍数", regular),
        (298, f"疑似过近阈值 {threshold:.2f} x", regular),
        (474, "R 提醒开关    P 暂停 / 恢复", regular),
        (506, "C 校准 / 重新校准    Q 或 Esc 退出", regular),
        (546, "全程静音 · 画面和关键点不保存", regular),
        (578, "倍数不是厘米距离，也不是医学判断", regular),
    )
    for y, label, font in labels:
        color = (149, 163, 184) if y >= 546 else (226, 232, 240)
        draw.text((20, y), label, font=font, fill=color)
    return panel


def draw_panel(
    frame, bounds, reading: Reading, config: Config, fonts,
    fps: float, latency: float | None, camera_info: str = "",
    reminder: ReminderView | None = None,
):
    import cv2
    import numpy as np
    from PIL import ImageDraw

    colors = {"正常": (87, 214, 164), "疑似过近": (255, 183, 77), "无法判断": (170, 184, 204)}
    color = colors[reading.state]
    regular, large = fonts
    panel = panel_background(fonts, config.near_ratio).copy()
    draw = ImageDraw.Draw(panel)
    reminder = reminder or ReminderView()
    if reminder.notice_visible:
        draw.rounded_rectangle((12, 324, 488, 394), radius=8, fill=(77, 48, 19), outline=(255, 183, 77), width=2)

    def text(y, value, fill=(226, 232, 240), font=regular, x=20):
        draw.text((x, y), value, font=font, fill=fill)

    def number(value, suffix=""):
        return "--" if value is None else f"{value:.2f}{suffix}"

    text(60, reading.state, fill=color, font=large)
    # 固定面板宽度，长中文原因换行，不遮挡摄像头画面。
    text(108, reading.reason[:22], fill=color)
    text(136, reading.reason[22:44], fill=color)
    text(182, number(reading.smooth_ratio, " x"), font=large, x=220)
    text(230, number(reading.raw_ratio, " x"), x=170)
    text(264, f"尺寸 {number(reading.current_size, ' px')}  /  基准 {number(reading.baseline_size, ' px')}")
    spread = "--" if reading.spread is None else f"{reading.spread * 100:.1f}%"
    text(298, f"波动 {spread}", x=300)
    if reading.progress is not None:
        text(332, f"校准进度  {reading.progress * 100:.0f}% / {config.calibration_seconds:g} 秒")
    else:
        text(332, f"连续过近  {reminder.elapsed:.1f} / {reminder.required_seconds:g} 秒")
    if reminder.notice_visible:
        text(366, "持续过近，请调整坐姿", fill=(255, 210, 128))
    elif reminder.paused:
        text(366, "提醒：暂停中，摄像头已释放")
    elif not reminder.enabled:
        text(366, "提醒：已关闭（仍显示检测数值）")
    elif reminder.cooldown > 0:
        text(366, f"提醒：开启  |  最早 {math.ceil(reminder.cooldown)} 秒后重复")
    else:
        text(366, "提醒：开启  |  连续达到时长后提示")
    inference = "--" if latency is None else f"{latency * 1000:.0f} ms"
    text(400, f"识别 {fps:.1f} 次/秒  |  推理 {inference}")
    text(434, camera_info, fill=(149, 163, 184))
    # 摄像头部分始终保持 OpenCV 的 BGR 格式，避免全画面反复转入/转出 Pillow。
    height, width = frame.shape[:2]
    canvas = np.empty((max(height, PANEL_HEIGHT), width + 500, 3), dtype=np.uint8)
    canvas[:height, :width] = cv2.flip(frame, 1)
    canvas[:PANEL_HEIGHT, width:] = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
    # 只填充画面和面板没有覆盖的余白，避免整幅画布重复写入。
    if height < PANEL_HEIGHT:
        canvas[height:, :width] = (39, 27, 20)
    elif height > PANEL_HEIGHT:
        canvas[PANEL_HEIGHT:, width:] = (39, 27, 20)
    if bounds is not None:
        left, top, right, bottom = bounds
        cv2.rectangle(canvas, (width - 1 - right, top), (width - 1 - left, bottom), color[::-1], 2)
    return canvas


def local_check(landmarker, config: Config, fonts):
    import mediapipe as mp
    import numpy as np

    blank = np.zeros((config.camera_height, config.camera_width, 3), dtype=np.uint8)
    # 完全合成的黑色图片，不读取真实摄像头、照片或用户文件。
    for timestamp in (0, 125, 250):
        result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=blank), timestamp)
        if result.face_landmarks:
            raise UserError("空白帧出现意外人脸结果，请检查模型。")
    draw_panel(blank, None, Reading("无法判断", "空白帧自检"), config, fonts, 0, 0)
    print("PASS: 本地模型加载、三次空白帧推理、中文面板绘制。未打开摄像头。")


def open_camera(args, config: Config):
    import cv2

    backend = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}[args.backend]
    # camera 参数限定为整数；不支持网络视频地址。
    capture = cv2.VideoCapture(args.camera, backend)
    try:
        if not capture.isOpened():
            raise UserError("无法打开摄像头。请关闭占用它的软件，检查相机权限或尝试 --camera 1 / --backend msmf。")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture
    except BaseException:
        capture.release()
        raise


def camera_loop(
    landmarker, args, config: Config, fonts,
    reminders: ReminderController | None = None,
    notice: SilentWindowNotice | None = None,
):
    import cv2
    import mediapipe as mp
    import numpy as np

    reminders = reminders or ReminderController(max_sample_gap=config.stale_seconds)
    capture = None
    try:
        capture = open_camera(args, config)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, config.camera_width + 500, max(config.camera_height, PANEL_HEIGHT))

        experiment = DistanceExperiment(config)
        next_detection = 0.0
        previous_timestamp = -1
        frame_shape = None
        inference_times = deque(maxlen=16)
        latency = None
        fps = 0.0
        capture_failed = False
        frame = np.zeros((config.camera_height, config.camera_width, 3), dtype=np.uint8)
        bounds = None
        reading = Reading("无法判断", "未校准，请按 C 保持正常坐姿 5 秒")
        camera_info = ""
        needs_redraw = True
        last_notice_shown = None
        while True:
            grabbed = False
            if not reminders.paused:
                read_started = time.monotonic()
                # 只在识别时刻解码；暂停期间完全不采集、不推理。
                grabbed = capture.grab()
                due = time.monotonic() >= next_detection
                if due or (not grabbed and not capture_failed):
                    ok, frame = capture.retrieve() if grabbed else (False, None)
                    captured_at = time.monotonic()
                    next_detection = captured_at + 1 / config.detection_fps
                    camera_info = (
                        f"相机索引 {args.camera} | {args.backend.upper()} | "
                        f"读帧 {(captured_at - read_started) * 1000:.0f} ms"
                    )
                    capture_failed = not ok or not is_color_frame(frame)
                    problem = frame_problem(frame if ok else None, config)
                    if problem:
                        if capture_failed:
                            frame = np.zeros((config.camera_height, config.camera_width, 3), dtype=np.uint8)
                        reading = experiment.update(captured_at, None, problem)
                        inference_times.clear()
                        fps = 0.0
                        latency = None
                        bounds = None
                    else:
                        if frame_shape is not None and frame.shape != frame_shape:
                            experiment = DistanceExperiment(config)
                            experiment.message = "摄像头分辨率已变化，请按 C 重新校准"
                        frame_shape = frame.shape
                        timestamp = max(previous_timestamp + 1, int(captured_at * 1000))
                        previous_timestamp = timestamp
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = landmarker.detect_for_video(
                            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp
                        )
                        latency = time.monotonic() - captured_at
                        if latency > config.stale_seconds:
                            measurement, bounds, reason = None, None, "推理耗时过长，数据已过期"
                        else:
                            measurement, bounds, reason = measure_face(
                                result.face_landmarks, frame.shape[1], frame.shape[0], config
                            )
                        reading = experiment.update(captured_at, measurement, reason)
                        inference_times.append(captured_at)
                        if len(inference_times) >= 2:
                            fps = (len(inference_times) - 1) / (inference_times[-1] - inference_times[0])
                        del result, rgb
                    reminders.update(reading.state, captured_at)
                    needs_redraw = True

            now = time.monotonic()
            reminder_view = reminders.view(now)
            if notice is not None:
                if reminder_view.notice_visible and reminders.last_notice_at != last_notice_shown:
                    if not notice.show(reminder_view.elapsed):
                        print("静音提示窗无法创建，请打开主窗口查看提醒。", file=sys.stderr)
                    last_notice_shown = reminders.last_notice_at
                elif not reminder_view.notice_visible:
                    notice.hide()
                notice.tick(now)

            # 最小化只停止预览绘制，识别、计时与独立提示窗仍照常运行。
            minimized = notice is not None and notice.main_is_minimized(WINDOW)
            if needs_redraw and not minimized:
                canvas = draw_panel(
                    frame, bounds, reading, config, fonts, fps, latency,
                    camera_info, reminder_view,
                )
                cv2.imshow(WINDOW, canvas)
                needs_redraw = False

            # 空闲时让出 CPU，同时继续响应按键和关闭事件。
            key = cv2.waitKey(20 if grabbed else 50) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                reminders.set_enabled(not reminders.enabled)
                needs_redraw = True
            if key in (ord("p"), ord("P")):
                reminders.set_paused(not reminders.paused)
                if reminders.paused:
                    capture.release()
                    capture = None
                    reading = experiment.invalidate("已暂停，摄像头已释放；按 P 恢复")
                    frame = np.zeros((config.camera_height, config.camera_width, 3), dtype=np.uint8)
                    bounds = None
                    camera_info = "摄像头已释放"
                else:
                    capture = open_camera(args, config)
                    capture_failed = False
                    reading = experiment.invalidate("正在恢复监测，请等待新画面")
                    camera_info = "摄像头已重新打开"
                inference_times.clear()
                fps = 0.0
                latency = None
                next_detection = 0.0
                needs_redraw = True
            if key in (ord("c"), ord("C")) and not reminders.paused:
                experiment.start_calibration(time.monotonic())
                reminders.reset_continuity()
                reading = Reading("无法判断", "校准中，请保持正常坐姿并正视屏幕", progress=0.0)
                needs_redraw = True
            if notice is not None:
                if not notice.main_window_exists(WINDOW):
                    break
            elif cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # 正常退出、窗口关闭、Ctrl+C 和推理异常都经过这里。
        if capture is not None:
            capture.release()
        if notice is not None:
            notice.hide()
        cv2.destroyAllWindows()


def main(argv=None):
    args, config = arguments(argv)
    if not args.model.is_file():
        print(f"缺少本地模型：{args.model}\n请参照 models/README.md 准备模型；程序不会联网下载。", file=sys.stderr)
        return 1
    # Matplotlib 是 MediaPipe 的间接依赖；缓存只含字体元数据。
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import cv2

        # 小尺寸像素转换不需要启用大量线程；此设置不改变 MediaPipe 的识别模型。
        cv2.setNumThreads(1)
        fonts = load_fonts()
        with create_landmarker(args.model, config) as landmarker:
            if args.check:
                local_check(landmarker, config, fonts)
            else:
                print("全程静音；后台和最小化时仍显示独立提示窗。C 校准，R 提醒开关，P 暂停/恢复，Q / Esc 退出。")
                reminders = ReminderController(
                    required_seconds=args.alert_after,
                    repeat_seconds=args.alert_interval,
                    enabled=not args.no_reminders,
                    max_sample_gap=config.stale_seconds,
                )
                camera_loop(landmarker, args, config, fonts, reminders, SilentWindowNotice())
    except ImportError:
        print("依赖未安装或无法导入。请使用项目 .venv，并按照 README.md 安装固定版本依赖。", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    except UserError as error:
        print(f"启动或运行失败：{error}", file=sys.stderr)
        return 1
    except Exception as error:
        # 未知第三方异常仅报告类型，不打印可能含输入内容的对象或完整追踪。
        print(f"程序已停止并释放摄像头（{type(error).__name__}）。请检查本地模型、相机驱动和运行环境。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
