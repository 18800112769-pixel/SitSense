"""禁用 Python 套接字的本地模型自检；只使用合成空白帧，不打开摄像头。

此检查不能替代用户断开所有网络后进行的真实摄像头验收，也不能审计原生 DLL 的全部行为。
"""

import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")


def deny_network(event, args):
    # platform.system() 在 Windows 上会查询本机名称；这不发送网络请求。
    # 其余套接字操作（包括 DNS、创建套接字和连接）一律拒绝。
    if event.startswith("socket.") and event != "socket.gethostname":
        raise RuntimeError("Offline check: Python socket access is disabled.")


sys.addaudithook(deny_network)

import socket

try:
    socket.socket()
except RuntimeError:
    print("PASS: Python socket guard is active.")
else:
    raise SystemExit("FAIL: Python socket guard was not effective.")

import app

raise SystemExit(app.main(["--check"]))
