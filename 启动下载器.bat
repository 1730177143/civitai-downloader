@echo off
chcp 65001 >nul
cd /d "%~dp0"

set PY=D:\AI\ComfyUI-aki-v3.2\python\pythonw.exe
if not exist "%PY%" set PY=D:\AI\ComfyUI-aki-v3.2\python\python.exe

start "" "%PY%" "%~dp0civitai_gui.py"
exit
