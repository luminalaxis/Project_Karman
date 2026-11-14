@echo off
chcp 65001 >nul
cd /d "%~dp0"
where pyw >nul 2>nul && (start "" pyw "launcher\karman_launcher.pyw" & exit /b)
where pythonw >nul 2>nul && (start "" pythonw "launcher\karman_launcher.pyw" & exit /b)
echo Python을 찾지 못했습니다. https://www.python.org 에서 설치한 뒤 다시 실행하세요.
pause
