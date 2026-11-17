# 本地 Face Landmarker 模型

模型已在用户授权后准备到本目录，文件名为 `face_landmarker.task`。
程序仅从本地文件读取，缺失或损坏时退出，不会自动下载。

- 官方说明：[Google MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- 模型来源：[Google 官方 Face Landmarker float16 / 版本 1](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
- SHA-256：`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`
- 准备日期：2026-11-16。
- 该模型用于生成本地关键点，本阶段不开启表情系数或姿态矩阵输出。

如果需要在另一台机器重新准备，可以复制此文件。也可以在明确同意联网准备后，单独执行下面的 PowerShell 命令。这是开发准备操作，不是 App 启动步骤；不使用摄像头。

```powershell
Set-Location 'D:\GitHub\SitSense'
Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task' -OutFile '.\models\face_landmarker.task'
Get-FileHash '.\models\face_landmarker.task' -Algorithm SHA256
```

校验值应与上方一致。该值用于复现这次获得的文件；不是额外的数字签名或安全认证。
