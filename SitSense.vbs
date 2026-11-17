' 无控制台启动源码版 SitSense（需要项目 .venv）。调试请用 run.bat。
Set fso = CreateObject("Scripting.FileSystemObject")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
Set shell = CreateObject("WScript.Shell")
shell.CurrentDirectory = dir
shell.Environment("Process")("PYTHONUTF8") = "1"
shell.Run """" & dir & "\.venv\Scripts\pythonw.exe"" -B """ & dir & "\desktop.py""", 0, False