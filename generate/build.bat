@echo off
gcc -O3 -o infer.exe infer.c -lm
if %errorlevel% == 0 (
    echo Build successful. Run with: infer.exe
) else (
    echo Build failed.
)
pause