@echo off
REM ================================================================
REM Run baseline inference on all 4 datasets with LLaVA-1.5-13B.
REM Execute this on the A100 machine after downloading datasets.
REM
REM Usage:
REM   run_all_baselines.bat
REM
REM Prerequisites:
REM   1. pip install -r requirements.txt
REM   2. Download all 4 datasets into data/
REM   3. Run python verify_loaders.py --data_root data/ to check
REM   4. Set proxy env vars if needed:
REM      set HTTP_PROXY=http://your-proxy:port
REM      set HTTPS_PROXY=http://your-proxy:port
REM ================================================================

echo ============================================
echo Running baselines with LLaVA-1.5-13B
echo ============================================

echo.
echo [1/4] HallusionBench...
python -m experiments.run_baseline --dataset hallusionbench --model llava --data_root data/ --output results/baselines/hallusionbench_llava.jsonl --mc_samples 5 --self_reported

echo.
echo [2/4] POPE...
python -m experiments.run_baseline --dataset pope --model llava --data_root data/ --output results/baselines/pope_llava.jsonl --mc_samples 5 --self_reported

echo.
echo [3/4] VSR...
python -m experiments.run_baseline --dataset vsr --model llava --data_root data/ --output results/baselines/vsr_llava.jsonl --mc_samples 5 --self_reported

echo.
echo [4/4] VizWiz...
python -m experiments.run_baseline --dataset vizwiz --model llava --data_root data/ --output results/baselines/vizwiz_llava.jsonl --mc_samples 5 --self_reported

echo.
echo ============================================
echo All baseline runs complete!
echo Results saved to results/baselines/
echo ============================================
pause
