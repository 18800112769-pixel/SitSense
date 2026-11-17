"""静音提醒规则：只记录当前进程内的计时状态，不保存历史、不发声。"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ReminderView:
    enabled: bool = True
    paused: bool = False
    elapsed: float = 0.0
    required_seconds: float = 30.0
    cooldown: float = 0.0
    notice_visible: bool = False


class ReminderController:
    def __init__(
        self, required_seconds: float = 30.0, repeat_seconds: float = 300.0,
        enabled: bool = True, max_sample_gap: float = 0.8,
    ):
        if not all(math.isfinite(value) and value > 0 for value in (required_seconds, repeat_seconds)):
            raise ValueError("持续提醒时间和重复间隔必须是有限的正数。")
        self.required_seconds = required_seconds
        self.repeat_seconds = repeat_seconds
        self.enabled = enabled
        self.paused = False
        self.max_sample_gap = max_sample_gap
        self.started_at: float | None = None
        self.last_sample_at: float | None = None
        self.last_notice_at: float | None = None
        self.notice_until = 0.0

    def reset_continuity(self):
        """恢复、失效或手动切换时清除连续计时，但不绕过已有重复提醒间隔。"""
        self.started_at = None
        self.last_sample_at = None
        self.notice_until = 0.0

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        self.reset_continuity()

    def set_paused(self, paused: bool):
        self.paused = paused
        self.reset_continuity()

    def update(self, state: str, now: float):
        # 采样停顿也必须打断连续异常，不能把阻塞、暂停或无法判断的时间算进去。
        if self.last_sample_at is not None:
            gap = now - self.last_sample_at
            if gap <= 0 or gap > self.max_sample_gap:
                self.reset_continuity()
        self.last_sample_at = now
        if not self.enabled or self.paused or state != "疑似过近":
            self.started_at = None
            self.notice_until = 0.0
            return
        if self.started_at is None:
            self.started_at = now
        elapsed = now - self.started_at
        can_repeat = self.last_notice_at is None or now - self.last_notice_at >= self.repeat_seconds
        if elapsed >= self.required_seconds and can_repeat:
            self.last_notice_at = now
            # 高亮提示保留 8 秒；恢复正常、失效或暂停时会提前清除。
            self.notice_until = now + 8.0

    def view(self, now: float) -> ReminderView:
        elapsed = 0.0 if self.started_at is None else max(0.0, now - self.started_at)
        cooldown = 0.0 if self.last_notice_at is None else max(
            0.0, self.repeat_seconds - (now - self.last_notice_at)
        )
        return ReminderView(
            enabled=self.enabled,
            paused=self.paused,
            elapsed=elapsed,
            required_seconds=self.required_seconds,
            cooldown=cooldown,
            notice_visible=self.enabled and not self.paused and now < self.notice_until,
        )


class AccumulatedReminder(ReminderController):
    """口部动作允许最多十秒状态跳变；这段宽限时间也计入异常。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_abnormal = None

    def reset_continuity(self):
        super().reset_continuity()
        self.last_abnormal = None

    def update(self, state, now):
        if self.last_sample_at is not None and not 0 < now - self.last_sample_at <= self.max_sample_gap:
            self.reset_continuity()
        self.last_sample_at = now
        if not self.enabled or self.paused:
            self.reset_continuity()
            return
        if self.last_abnormal is not None and now - self.last_abnormal > 10.0:
            self.reset_continuity()
            self.last_sample_at = now
        if state == "疑似过近":
            self.last_abnormal = now
            if self.started_at is None:
                self.started_at = now
        if self.started_at is None:
            return
        if now - self.started_at > self.required_seconds and (
            self.last_notice_at is None or now - self.last_notice_at >= self.repeat_seconds
        ):
            self.last_notice_at = now
            self.notice_until = now + 8.0
