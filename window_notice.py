"""Windows 原生静音提示窗：独立于主窗口，置顶显示但不激活、不抢键盘焦点。"""

import ctypes
from ctypes import wintypes
import time


class MonitorInfo(ctypes.Structure):
    _fields_ = (
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    )


def bind(library, name, result, *arguments):
    function = getattr(library, name)
    function.restype = result
    function.argtypes = list(arguments)
    return function


class SilentWindowNotice:
    def __init__(self):
        # 明确声明 64 位指针类型，避免窗口句柄被 ctypes 默认的 int 截断。
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
        self.create_window = bind(
            self.user32, "CreateWindowExW", wintypes.HWND,
            wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            wintypes.HWND, wintypes.HANDLE, wintypes.HINSTANCE, wintypes.LPVOID,
        )
        self.destroy_window = bind(self.user32, "DestroyWindow", wintypes.BOOL, wintypes.HWND)
        self.show_window = bind(self.user32, "ShowWindow", wintypes.BOOL, wintypes.HWND, ctypes.c_int)
        self.update_window = bind(self.user32, "UpdateWindow", wintypes.BOOL, wintypes.HWND)
        self.set_position = bind(
            self.user32, "SetWindowPos", wintypes.BOOL,
            wintypes.HWND, wintypes.HWND,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT,
        )
        self.foreground = bind(self.user32, "GetForegroundWindow", wintypes.HWND)
        self.find_window = bind(self.user32, "FindWindowW", wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR)
        self.is_window = bind(self.user32, "IsWindow", wintypes.BOOL, wintypes.HWND)
        self.is_iconic = bind(self.user32, "IsIconic", wintypes.BOOL, wintypes.HWND)
        self.monitor_from_window = bind(
            self.user32, "MonitorFromWindow", wintypes.HANDLE, wintypes.HWND, wintypes.DWORD
        )
        self.monitor_info = bind(
            self.user32, "GetMonitorInfoW", wintypes.BOOL,
            wintypes.HANDLE, ctypes.POINTER(MonitorInfo),
        )
        self.system_parameters = bind(
            self.user32, "SystemParametersInfoW", wintypes.BOOL,
            wintypes.UINT, wintypes.UINT, wintypes.LPVOID, wintypes.UINT,
        )
        self.system_metrics = bind(self.user32, "GetSystemMetrics", ctypes.c_int, ctypes.c_int)
        self.get_dpi = bind(self.user32, "GetDpiForWindow", wintypes.UINT, wintypes.HWND)
        self.send_message = bind(
            self.user32, "SendMessageW", ctypes.c_ssize_t,
            wintypes.HWND, wintypes.UINT, ctypes.c_size_t, ctypes.c_ssize_t,
        )
        self.peek_message = bind(
            self.user32, "PeekMessageW", wintypes.BOOL,
            ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT,
        )
        self.translate_message = bind(self.user32, "TranslateMessage", wintypes.BOOL, ctypes.POINTER(wintypes.MSG))
        self.dispatch_message = bind(self.user32, "DispatchMessageW", ctypes.c_ssize_t, ctypes.POINTER(wintypes.MSG))
        self.create_font = bind(
            self.gdi32, "CreateFontW", wintypes.HANDLE,
            *([ctypes.c_int] * 5), *([wintypes.DWORD] * 8), wintypes.LPCWSTR,
        )
        self.delete_object = bind(self.gdi32, "DeleteObject", wintypes.BOOL, wintypes.HANDLE)
        self.handle = None
        self.main_handle = None
        self.fonts = []
        self.deadline = 0.0

    def main_window_exists(self, title: str) -> bool:
        if not self.main_handle or not self.is_window(self.main_handle):
            self.main_handle = self.find_window(None, title)
        return bool(self.main_handle and self.is_window(self.main_handle))

    def main_is_minimized(self, title: str) -> bool:
        return self.main_window_exists(title) and bool(self.is_iconic(self.main_handle))

    def hide(self):
        if self.handle:
            self.destroy_window(self.handle)
            self.handle = None
        for font in self.fonts:
            self.delete_object(font)
        self.fonts.clear()
        self.deadline = 0.0

    def show(self, elapsed: float, seconds: float = 8.0) -> bool:
        self.hide()
        foreground = self.foreground()
        monitor = self.monitor_from_window(foreground, 2)  # 最近的工作屏幕
        info = MonitorInfo()
        info.cbSize = ctypes.sizeof(MonitorInfo)
        if self.monitor_info(monitor, ctypes.byref(info)):
            work = info.rcWork
        else:
            work = wintypes.RECT()
            if not self.system_parameters(0x0030, 0, ctypes.byref(work), 0):
                work = wintypes.RECT(0, 0, self.system_metrics(0), self.system_metrics(1))
        scale = max(1.0, self.get_dpi(foreground) / 96.0)
        margin = round(20 * scale)
        width = min(round(420 * scale), work.right - work.left - margin * 2)
        height = min(round(180 * scale), work.bottom - work.top - margin * 2)
        x = work.right - width - margin
        y = work.bottom - height - margin

        # 不设置主窗口为 owner；否则主窗口最小化时，提示窗也会一起隐藏。
        # WS_EX_TOPMOST | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
        self.handle = self.create_window(
            0x00000008 | 0x08000000 | 0x00000080,
            "STATIC", "", 0x80000000 | 0x00800000 | 0x02000000,
            x, y, width, height, None, None, None, None,
        )
        if not self.handle:
            return False
        labels = (
            (18, 34, 22, 600, "SitSense · 坐姿提醒"),
            (60, 62, 20, 400, f"距离屏幕过近，请调整坐姿。\n已连续过近 {elapsed:.0f} 秒。"),
            (135, 26, 16, 400, f"静音提示 · 约 {seconds:g} 秒后自动关闭"),
        )
        for top, label_height, size, weight, text in labels:
            child = self.create_window(
                0, "STATIC", text, 0x40000000 | 0x10000000 | 0x00000080,
                margin, round(top * scale), width - margin * 2, round(label_height * scale),
                self.handle, None, None, None,
            )
            if not child:
                self.hide()
                return False
            font = self.create_font(
                -round(size * scale), 0, 0, 0, weight,
                0, 0, 0, 1, 0, 0, 5, 0, "Microsoft YaHei",
            )
            if font:
                self.fonts.append(font)
                self.send_message(child, 0x0030, font, 1)  # WM_SETFONT
        self.show_window(self.handle, 4)  # SW_SHOWNOACTIVATE
        self.set_position(self.handle, ctypes.c_void_p(-1), x, y, width, height, 0x0010 | 0x0040)
        self.update_window(self.handle)
        self.deadline = time.monotonic() + seconds
        return True

    def tick(self, now: float):
        if not self.handle:
            return
        if now >= self.deadline:
            self.hide()
            return
        # 只分发自身窗口消息，避免吞掉 OpenCV 主窗口的控制键。
        message = wintypes.MSG()
        while self.peek_message(ctypes.byref(message), self.handle, 0, 0, 1):
            self.translate_message(ctypes.byref(message))
            self.dispatch_message(ctypes.byref(message))
