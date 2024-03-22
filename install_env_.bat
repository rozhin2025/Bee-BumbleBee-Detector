@echo off
call mamba create -n rozhin python=3.10 -y
call mamba activate rozhin
call mamba install -c conda-forge cudatoolkit=11.2 cudnn=8.2.1 cupy=10.1.0 -y
call python -m pip install "tensorflow<2.11"
call mamba install scipy librosa ipykernel pyfftw seaborn matplotlib xgboost -y
call mamba install -c nvidia cuda-nvcc -y
call mamba deactivate
pause
