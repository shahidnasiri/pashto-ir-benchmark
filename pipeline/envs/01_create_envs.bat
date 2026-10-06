@echo off
REM ============================================================
REM Pashto IR Benchmark - Local Environment Setup (Windows + conda)
REM Run this from Anaconda Prompt, from anywhere.
REM Creates 3 isolated environments matching Section 4.2 of the paper.
REM ============================================================

echo [1/3] Creating 'pashto-main' environment (8 of 10 models)...
call conda create -n pashto-main python=3.11 -y
call conda activate pashto-main
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements_main.txt
call conda deactivate

echo [2/3] Creating 'pashto-jina-gte' environment (jina-v3 + gte-multilingual)...
call conda create -n pashto-jina-gte python=3.11 -y
call conda activate pashto-jina-gte
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements_jina_gte.txt
call conda deactivate

echo [3/3] Creating 'pashto-nomic' environment (nomic-embed-text-v2-moe)...
call conda create -n pashto-nomic python=3.11 -y
call conda activate pashto-nomic
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements_nomic.txt
call conda deactivate

echo.
echo ============================================================
echo Done. Verify each env with:
echo   conda activate pashto-main      && python verify_gpu.py
echo   conda activate pashto-jina-gte  && python verify_gpu.py
echo   conda activate pashto-nomic     && python verify_gpu.py
echo ============================================================
pause
