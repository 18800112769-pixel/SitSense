"""SitSense 本地个人校准数据存储。

只保存姿态检测所需的数值基准，不保存图像、视频、关键点或面部特征序列。
"""

from datetime import datetime
import json
import math
import os
from pathlib import Path
from typing import Any


CALIBRATION_VERSION = 1


class CalibrationStore:
    def __init__(self, path, *, camera: int, backend: str, frame_width: int, frame_height: int):
        self.path = Path(path)
        self.context = {
            "camera": int(camera),
            "backend": str(backend),
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
        }
        self.posture_baseline = None
        self.mouth_baseline = None
        self.status = "尚无已保存的校准数据"
        self.load()

    @staticmethod
    def _valid_posture(value):
        if isinstance(value, bool):
            return False
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(number) and number > 0

    @staticmethod
    def _valid_mouth(values):
        if not isinstance(values, (list, tuple)) or len(values) != 5:
            return False
        try:
            numbers = tuple(float(value) for value in values)
        except (TypeError, ValueError):
            return False
        return (
            all(math.isfinite(value) for value in numbers)
            and all(0 <= value <= 1 for value in numbers[:3])
            and all(value > 0 for value in numbers[3:])
        )

    def load(self):
        """只在相机配置与保存时一致时加载，避免误用不同摄像头的像素基准。"""
        if not self.path.is_file():
            return False
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            self.status = "本地校准文件无法读取，请重新校准"
            return False
        if not isinstance(payload, dict) or payload.get("version") != CALIBRATION_VERSION:
            self.status = "本地校准文件版本不兼容，请重新校准"
            return False
        if any(payload.get(key) != value for key, value in self.context.items()):
            self.status = "摄像头配置已变化，需要重新校准"
            return False

        posture = payload.get("posture_baseline_size")
        mouth = payload.get("mouth_baseline_features")
        if self._valid_posture(posture):
            self.posture_baseline = float(posture)
        if self._valid_mouth(mouth):
            self.mouth_baseline = tuple(float(value) for value in mouth)

        if self.posture_baseline is not None or self.mouth_baseline is not None:
            self.status = "已加载本地校准数据"
            return True
        self.status = "校准文件中没有有效基准，请重新校准"
        return False

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version": CALIBRATION_VERSION,
            **self.context,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "posture_baseline_size": self.posture_baseline,
            "mouth_baseline_features": (
                list(self.mouth_baseline) if self.mouth_baseline is not None else None
            ),
        }
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
            self.status = "校准数据已保存在本机"
        finally:
            # 写入或替换失败时清理临时文件；旧的正式文件保持不动。
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def save_posture(self, baseline):
        if not self._valid_posture(baseline):
            raise ValueError("坐姿校准基准无效，拒绝保存。")
        self.posture_baseline = float(baseline)
        self._save()

    def save_mouth(self, baseline):
        if not self._valid_mouth(baseline):
            raise ValueError("口部校准基准无效，拒绝保存。")
        self.mouth_baseline = tuple(float(value) for value in baseline)
        self._save()
