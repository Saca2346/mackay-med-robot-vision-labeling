@echo off
rem Run a server step FROM THE LAPTOP over ssh (no Remote Desktop, no copy-paste).
rem Usage:  RUN_ON_SERVER.bat NAME      runs server_scripts\NAME.sh on the server, e.g. langkah7_manifest
rem The server output is shown here AND saved to server_results\NAME.txt, so Claude can read it directly.
rem Without an SSH key you are asked for the server password (twice per run). Nothing is stored.
rem With the key of docs\REMOTE_SERVER_CLAUDE_CODEX.md (code repo) no password is asked.
cd /d "%~dp0"
if "%1"=="" goto usage
if not exist server_scripts\%1.sh goto missing
if not exist server_results mkdir server_results
scp server_scripts\%1.sh Zen@140.113.149.94:~/%1.sh
ssh Zen@140.113.149.94 "sed -i 's/\r$//' ~/%1.sh && bash ~/%1.sh" > server_results\%1.txt 2>&1
type server_results\%1.txt
echo.
echo saved: %~dp0server_results\%1.txt
exit /b 0
:usage
echo usage: RUN_ON_SERVER.bat NAME   (a file server_scripts\NAME.sh, e.g. langkah7_manifest)
exit /b 1
:missing
echo server_scripts\%1.sh not found
exit /b 1
