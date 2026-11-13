@echo off
chcp 65001 >nul
rem Project_Karman 폴더 아이콘 적용 (clone 후 한 번 실행)
cd /d "%~dp0.."
attrib +s +h desktop.ini
attrib +r "%cd%"
ie4uinit.exe -show
echo 폴더 아이콘을 적용했습니다. 바로 안 바뀌면 탐색기를 새로고침(F5)하세요.
pause
