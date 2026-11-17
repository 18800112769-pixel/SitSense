"""阶段一的纯计算规则。只在内存保留短窗口数据，不涉及摄像头、文件或网络。"""

from collections import deque
from dataclasses import dataclass
import math
from statistics import median


# MediaPipe 的脸部外轮廓；不使用虹膜、嘴部表情或任何身份特征。
FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)


@dataclass(frozen=True)
class Config:
    calibration_seconds: float = 5.0
    near_ratio: float = 1.15
    detection_fps: float = 5.0
    smoothing_seconds: float = 0.6
    minimum_samples: int = 3
    stale_seconds: float = 0.8
    calibration_min_coverage: float = 0.8
    calibration_max_spread: float = 0.06
    live_max_spread: float = 0.10
    min_face_pixels: float = 60.0
    frame_margin: float = 0.02
    detection_confidence: float = 0.7
    presence_confidence: float = 0.7
    tracking_confidence: float = 0.6
    camera_width: int = 640
    camera_height: int = 480
    dark_pixel_threshold: float = 12.0
    dark_percentile: float = 95.0

    def __post_init__(self):
        positive = (
            self.calibration_seconds, self.near_ratio, self.detection_fps,
            self.smoothing_seconds, self.stale_seconds, self.min_face_pixels,
        )
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("实验参数必须是有限的正数。")
        if self.near_ratio <= 1:
            raise ValueError("过近阈值必须大于 1。")
        if not 5 <= self.detection_fps <= 10:
            raise ValueError("阶段一的识别频率限定为 5～10 次/秒。")
        if self.calibration_seconds < 3:
            raise ValueError("校准时间至少为 3 秒，建议保持默认 5 秒。")
        if self.smoothing_seconds * self.detection_fps < self.minimum_samples:
            raise ValueError("平滑窗口太短，无法积累足够样本。")


@dataclass(frozen=True)
class Measurement:
    width: float
    height: float

    @property
    def size(self) -> float:
        return math.sqrt(self.width * self.height)


@dataclass(frozen=True)
class Reading:
    state: str
    reason: str
    raw_ratio: float | None = None
    smooth_ratio: float | None = None
    current_size: float | None = None
    baseline_size: float | None = None
    progress: float | None = None
    spread: float | None = None


def measure_face(faces, frame_width: int, frame_height: int, config: Config):
    """提取像素尺寸和绘图边界。输出不代表物理距离或可靠的头部角度。"""
    if not faces:
        return None, None, "未检测到人脸"
    if len(faces) != 1:
        return None, None, "画面中有多张人脸，请单人测试"
    landmarks = faces[0]
    if len(landmarks) <= max(FACE_OVAL):
        return None, None, "人脸关键点不足"
    points = [(landmarks[index].x, landmarks[index].y) for index in FACE_OVAL]
    if not all(math.isfinite(value) for point in points for value in point):
        return None, None, "人脸关键点无效"
    left = min(point[0] for point in points)
    right = max(point[0] for point in points)
    top = min(point[1] for point in points)
    bottom = max(point[1] for point in points)
    margin = config.frame_margin
    if left < margin or top < margin or right > 1 - margin or bottom > 1 - margin:
        return None, None, "人脸靠近画面边缘，轮廓可能不完整"
    measurement = Measurement((right - left) * frame_width, (bottom - top) * frame_height)
    if min(measurement.width, measurement.height) < config.min_face_pixels:
        return None, None, "人脸太小，暂时无法可靠测量"
    bounds = (
        round(left * frame_width), round(top * frame_height),
        round(right * frame_width), round(bottom * frame_height),
    )
    return measurement, bounds, ""


def relative_spread(values) -> float:
    """极差用于保守拒绝抖动；中位数用于减小短时间测量波动。"""
    return (max(values) - min(values)) / median(values)


class DistanceExperiment:
    def __init__(self, config: Config):
        self.config = config
        self.baseline: float | None = None
        self.previous_baseline: float | None = None
        self.calibration_start: float | None = None
        self.samples: list[tuple[float, Measurement]] = []
        self.recent: deque[tuple[float, Measurement]] = deque()
        self.last_time: float | None = None
        self.message = "未校准，请按 C 保持正常坐姿 5 秒"

    def load_baseline(self, baseline: float) -> bool:
        """加载本地保存的个人基准；只接受有限的正数。"""
        if isinstance(baseline, bool):
            return False
        try:
            value = float(baseline)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(value) or value <= 0:
            return False
        self.baseline = value
        self.previous_baseline = None
        self.calibration_start = None
        self.samples.clear()
        self.recent.clear()
        self.last_time = None
        self.message = "已加载本地坐姿校准基准"
        return True

    def start_calibration(self, now: float):
        # 新校准成功前暂时停用旧基准；失败时恢复旧基准，避免一次失败让已校准用户失去监测。
        self.previous_baseline = self.baseline
        self.baseline = None
        self.calibration_start = now
        self.samples.clear()
        self.recent.clear()
        self.message = "校准中，请保持正常坐姿并正视屏幕"

    def _fail_calibration(self, reason: str) -> Reading:
        self.calibration_start = None
        self.samples.clear()
        if self.previous_baseline is not None:
            self.baseline = self.previous_baseline
            self.message = f"校准失败，继续使用上次校准基准：{reason}"
        else:
            self.baseline = None
            self.message = f"校准失败：{reason}；请按 C 重试"
        self.previous_baseline = None
        self.recent.clear()
        return self._unknown(self.message)

    def _unknown(self, reason: str, **values) -> Reading:
        return Reading("无法判断", reason, baseline_size=self.baseline, **values)

    def invalidate(self, reason: str) -> Reading:
        self.recent.clear()
        if self.calibration_start is not None:
            return self._fail_calibration(reason)
        return self._unknown(reason)

    def _finish_calibration(self) -> Reading:
        config = self.config
        samples = self.samples
        self.samples = []
        self.calibration_start = None
        minimum_count = math.ceil(
            config.calibration_seconds * config.detection_fps * config.calibration_min_coverage
        )
        enough_time = (
            len(samples) >= 2
            and samples[-1][0] - samples[0][0]
            >= config.calibration_seconds * config.calibration_min_coverage
        )
        if len(samples) < minimum_count or not enough_time:
            return self._fail_calibration("有效采样不足；请关闭高占用程序后重试")
        # 同时检查宽和高，避免一增一减时几何平均值看似稳定。
        width_spread = relative_spread([sample.width for _, sample in samples])
        height_spread = relative_spread([sample.height for _, sample in samples])
        if max(width_spread, height_spread) > config.calibration_max_spread:
            return self._fail_calibration("人脸尺寸变化过大；请坐稳后重试")
        self.baseline = median([sample.size for _, sample in samples])
        self.previous_baseline = None
        self.recent.clear()
        self.message = "校准成功，正在积累稳定数据"
        return self._unknown(self.message)

    def update(self, now: float, measurement: Measurement | None, reason: str = "") -> Reading:
        if not math.isfinite(now):
            return self.invalidate("采样时间无效")
        interrupted = False
        if self.last_time is not None:
            if now <= self.last_time:
                return self.invalidate("采样时间未递增")
            interrupted = now - self.last_time > self.config.stale_seconds
        self.last_time = now
        # 先保留当前帧的明确故障，再处理采样间隔；不能用“采样中断”掩盖黑屏或丢脸。
        if measurement is None:
            return self.invalidate(reason or "未获得可靠的人脸数据")
        if not all(
            math.isfinite(value) and value > 0
            for value in (measurement.width, measurement.height)
        ):
            return self.invalidate("人脸尺寸无效")
        if interrupted:
            return self.invalidate("采样中断，请等待新的稳定数据")
        if self.calibration_start is not None:
            self.samples.append((now, measurement))
            elapsed = now - self.calibration_start
            if elapsed >= self.config.calibration_seconds:
                return self._finish_calibration()
            return self._unknown(
                self.message,
                progress=max(0.0, elapsed / self.config.calibration_seconds),
                current_size=measurement.size,
            )
        if self.baseline is None:
            return self._unknown(self.message, current_size=measurement.size)

        self.recent.append((now, measurement))
        while self.recent and now - self.recent[0][0] > self.config.smoothing_seconds:
            self.recent.popleft()
        values = dict(raw_ratio=measurement.size / self.baseline, current_size=measurement.size)
        if len(self.recent) < self.config.minimum_samples:
            return self._unknown("正在积累稳定数据", **values)
        spread = max(
            relative_spread([sample.width for _, sample in self.recent]),
            relative_spread([sample.height for _, sample in self.recent]),
        )
        values["spread"] = spread
        if spread > self.config.live_max_spread:
            return self._unknown("尺寸变化较快，请稍停片刻", **values)
        smoothed = median([sample.size for _, sample in self.recent]) / self.baseline
        values["smooth_ratio"] = smoothed
        near = smoothed >= self.config.near_ratio
        return Reading(
            "疑似过近" if near else "正常",
            "人脸尺寸达到实验阈值" if near else "人脸尺寸未达到实验阈值",
            baseline_size=self.baseline,
            **values,
        )
