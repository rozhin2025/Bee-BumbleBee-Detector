@echo off
call mamba create --solver=libmamba -n thesis -c rapidsai -c conda-forge -c nvidia rapids=24.02 python=3.10 cuda-version=12.0 tensorflow -y
call mamba install -c conda-forge scipy librosa ipykernel pyfftw seaborn matplotlib xgboost -y
call mamba deactivate
pause
