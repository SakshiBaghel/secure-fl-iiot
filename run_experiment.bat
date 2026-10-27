@echo off
title Secure FL for IIoT - Federated Learning Pipeline
echo ========================================================
echo   Running Federated Learning Experiment & Security Defense
echo ========================================================
echo.
python src/run_experiment.py --rounds 5 --epochs 3 --threshold 15.0
echo.
echo ========================================================
echo   Experiment Finished! Results saved to results/
echo ========================================================
pause
