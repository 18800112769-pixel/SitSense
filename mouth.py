"""简化口部检测：共享人脸推理，个人基准 + 多指标 + 持续计时。

仅判断可见的嘴部变化，不能确认撕皮或咬腮。逐帧特征只驻留内存；个人校准基准由桌面端单独保存到本机。
"""

from collections import deque
from statistics import median
import math


class MouthMonitor:
    def __init__(self):
        self.baseline = None
        self.previous_baseline = None
        self.calibrating = None
        self.samples = []
        self.recent = deque()
        self.last_at = None
        self.bad_since = None
        self.active = False
        self.reason = "请放松嘴唇后校准"

    def load_baseline(self, baseline):
        """加载本地保存的个人口部基准；格式错误时拒绝加载。"""
        if not isinstance(baseline, (list, tuple)) or len(baseline) != 5:
            return False
        try:
            values = tuple(float(value) for value in baseline)
        except (TypeError, ValueError):
            return False
        if not all(math.isfinite(value) for value in values):
            return False
        if any(not 0 <= value <= 1 for value in values[:3]) or any(value <= 0 for value in values[3:]):
            return False
        self.baseline = values
        self.previous_baseline = None
        self.calibrating = None
        self.samples.clear()
        self.reset()
        self.reason = "已加载本地口部校准基准"
        return True

    def calibrate(self, now):
        # 新校准成功前暂时停用旧基准；失败时恢复上次有效基准。
        self.previous_baseline = self.baseline
        self.baseline = None
        self.calibrating = now
        self.samples.clear()
        self.reset()

    def reset(self):
        self.recent.clear()
        self.last_at = None
        self.bad_since = None
        self.active = False

    def skip(self, reason, now):
        # 校准中的偶发坏帧只跳过不中断；连续坏帧超过 2 秒才判定校准失败。
        if self.calibrating is None:
            return self.unknown(reason)
        self.last_at = now
        if self.bad_since is None:
            self.bad_since = now
        if now - self.bad_since > 2.0:
            return self.unknown(reason)
        self.reason = f"校准中，等待稳定画面：{reason}"
        return "无法判断"

    def _stable(self):
        # 去掉每列最高和最低各 10% 的离群样本后再看极差，避免单帧抖动毁掉整次校准。
        if len(self.samples) < 18:
            return False
        cut = max(1, len(self.samples) // 10)
        for i in range(5):
            column = sorted(row[i] for row in self.samples)[cut:-cut]
            if column[-1] - column[0] >= 0.15:
                return False
        return True

    def unknown(self, reason):
        self.reset()
        if self.calibrating is not None:
            self.calibrating = None
            self.samples.clear()
            if self.previous_baseline is not None:
                self.baseline = self.previous_baseline
                reason = f"口部校准中断，继续使用上次校准基准：{reason}"
            else:
                self.baseline = None
                reason = "校准中断，请放松正视后重试"
            self.previous_baseline = None
        self.reason = reason
        return "无法判断"

    def update(self, result, now):
        if self.last_at is not None and not 0 < now - self.last_at <= 0.8:
            return self.unknown("采样中断，等待稳定画面")
        self.last_at = now
        if len(result.face_landmarks) != 1 or len(result.face_blendshapes) != 1:
            return self.skip("需要清晰的单人正脸", now)
        points = result.face_landmarks[0]
        scores = {item.category_name: item.score for item in result.face_blendshapes[0]}
        names = ("mouthPressLeft", "mouthPressRight", "mouthRollLower", "mouthRollUpper",
                 "mouthPucker", "mouthFunnel", "jawOpen", "mouthSmileLeft", "mouthSmileRight")
        if len(points) < 292 or any(name not in scores for name in names):
            return self.skip("口部数据不足", now)
        values = [scores[name] for name in names]
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
            return self.skip("口部数据不稳定", now)
        eye_span = math.hypot(points[263].x - points[33].x, points[263].y - points[33].y)
        if eye_span < 0.03:
            return self.skip("人脸过小", now)
        # 只做简单的正脸门限；不尝试识别复杂转头或被手遮挡的行为。
        nose_offset = abs(points[1].x - (points[33].x + points[263].x) / 2) / eye_span
        if nose_offset > 0.24 or abs(points[33].y - points[263].y) / eye_span > 0.25:
            return self.skip("请正视屏幕", now)
        if scores["jawOpen"] > 0.16 or max(scores["mouthSmileLeft"], scores["mouthSmileRight"]) > 0.35:
            return self.skip("张嘴或表情变化，暂不判断", now)
        width = math.hypot(points[291].x - points[61].x, points[291].y - points[61].y) / eye_span
        thickness = math.hypot(points[0].x - points[17].x, points[0].y - points[17].y) / eye_span
        features = (max(values[0:2]), max(values[2:4]), max(values[4:6]), width, thickness)
        if not all(math.isfinite(value) for value in features):
            return self.skip("口部定位不稳定", now)
        if self.calibrating is not None:
            self.bad_since = None
            self.samples.append(features)
            elapsed = now - self.calibrating
            self.reason = f"放松嘴唇校准中 {min(5, elapsed):.0f} / 5 秒"
            if elapsed >= 5:
                if self._stable():
                    self.baseline = tuple(median(row[i] for row in self.samples) for i in range(5))
                    self.previous_baseline = None
                    self.reason = "校准完成"
                elif elapsed < 8:
                    # 前 5 秒不够稳定时延长到 8 秒，而不是立即失败。
                    self.reason = f"请继续保持放松 {elapsed:.0f} / 8 秒"
                    return "无法判断"
                elif self.previous_baseline is not None:
                    self.baseline = self.previous_baseline
                    self.previous_baseline = None
                    self.reason = "口部校准失败，继续使用上次校准基准"
                else:
                    self.baseline = None
                    self.reason = "口部变化较大，请重新校准"
                self.samples.clear()
                self.calibrating = None
            return "无法判断"
        if self.baseline is None:
            self.reason = "请放松嘴唇后校准"
            return "无法判断"
        self.recent.append((now, features))
        while self.recent and now - self.recent[0][0] > 0.7:
            self.recent.popleft()
        if len(self.recent) < 3:
            self.reason = "等待稳定数据"
            return "无法判断"
        smooth = tuple(median(row[i] for _, row in self.recent) for i in range(5))
        delta = [smooth[i] - self.baseline[i] for i in range(5)]
        # 表情系数与嘴形必须同时变化；恢复门限稍低，减少状态抖动。
        factor = 0.7 if self.active else 1.0
        self.active = (
            (max(delta[:2]) > 0.20 * factor and delta[4] < -0.025 * factor)
            or (delta[2] > 0.25 * factor and delta[3] < -0.07 * factor)
        )
        self.reason = "持续嘴部变化" if self.active else "嘴部放松"
        return "疑似异常" if self.active else "正常"
