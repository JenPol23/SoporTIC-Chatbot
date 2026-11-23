@echo off
rem Inicia SoporTIC con los modelos guardados (sin entrenar) y abre la interfaz en http://127.0.0.1:7860
rem (si el puerto esta ocupado, usa el siguiente libre y lo avisa)
rem Respaldo de emergencia si el modo IA va lento: iniciar_soportic.bat 1.5B  (usa Qwen2.5-1.5B en lugar del 3B)
chcp 65001 >nul
if /i "%~1"=="1.5B" set SOPORTIC_GENERADOR=1.5B
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo No se encontro el entorno .venv en %CD%.
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
python iniciar_soportic.py
if errorlevel 1 (
    echo.
    echo SoporTIC se detuvo con un error. Revise el mensaje anterior.
)
pause
