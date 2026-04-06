@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d C:\claude\wave-engine
cargo build --features candle-backend 2>&1 > C:\claude\wave-engine\build_output.txt 2>&1
echo BUILD_EXIT_CODE=%ERRORLEVEL% >> C:\claude\wave-engine\build_output.txt
