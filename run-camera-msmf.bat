@echo off
call "%~dp0run.bat" --backend msmf %*
exit /b %errorlevel%
