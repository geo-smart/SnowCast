#!/bin/bash

# source /home/chetana/miniconda/bin/activate
# pip install geojson seaborn xarray dask pandas numpy geopandas pyarrow netCDF4 h5netcdf torch scikit-learn psutil

# # mkdir -p /home/chetana/data/ucla/raw/

# # cd /home/chetana/data/ucla/raw/

# # wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies --no-check-certificate --auth-no-challenge=on -r --reject "index.html*" -np -e robots=off https://n5eil01u.ecs.nsidc.org/DP6/SNOWEX/WUS_UCLA_SR.001/1984.10.01/WUS_UCLA_SR_v01_N31_0W104_0_agg_16_WY1984_85_SWE_SCA_POST.nc

# # wget https://n5eil01u.ecs.nsidc.org/DP6/SNOWEX/WUS_UCLA_SR.001/1984.10.01/WUS_UCLA_SR_v01_N31_0W104_0_agg_16_WY1984_85_SWE_SCA_POST.nc

# exit 0

# change this line if you are creating a new env
# source /home/ubuntu/anaconda3/bin/activate
# conda activate new_env
# which python
# python --version

# install GDAL
# sudo apt update
# sudo apt install gdal-bin libgdal-dev -y

# Use cat to create the requirements.txt file
cat <<EOL > /home/chetana/requirements.txt
affine==2.4.0
attrs==25.1.0
certifi==2025.1.31
charset-normalizer==3.4.1
click==8.1.8
click-plugins==1.1.1
cligj==0.7.2
cloudpickle==3.0.0
contourpy==1.3.0
cycler==0.12.1
filelock==3.17.0
fonttools==4.55.8
fsspec==2025.2.0
geojson==3.2.0
geopandas==1.0.1
idna==3.10
importlib_metadata==8.6.1
importlib_resources==6.5.2
Jinja2==3.1.5
joblib==1.4.2
kiwisolver==1.4.7
llvmlite==0.43.0
MarkupSafe==3.0.2
matplotlib==3.9.4
mpmath==1.3.0
networkx==3.2.1
numba==0.60.0
numpy==1.26.4
nvidia-cublas-cu11==11.10.3.66
nvidia-cublas-cu12==12.4.5.8
nvidia-cuda-cupti-cu12==12.4.127
nvidia-cuda-nvrtc-cu11==11.7.99
nvidia-cuda-nvrtc-cu12==12.4.127
nvidia-cuda-runtime-cu11==11.7.99
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cudnn-cu11==8.5.0.96
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.1.3
nvidia-curand-cu12==10.3.5.147
nvidia-cusolver-cu12==11.6.1.9
nvidia-cusparse-cu12==12.3.1.170
nvidia-cusparselt-cu12==0.6.2
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu12==12.4.127
packaging==24.2
pandas==2.2.2
pillow==11.1.0
psutil==6.1.1
pyogrio==0.10.0
pyparsing==3.2.1
pyproj==3.6.1
python-dateutil==2.9.0.post0
pytorch-tabnet==4.1.0
pytz==2024.1
rasterio==1.4.3
requests==2.32.3
scikit-learn==1.6.1
scipy==1.13.1
seaborn==0.13.2
shap==0.46.0
shapely==2.0.7
six==1.16.0
slicer==0.0.8
sympy==1.13.1
threadpoolctl==3.5.0
torch==1.13.1
torchaudio==0.13.1
torchmetrics==0.11.1
torchvision==0.14.1
tqdm==4.67.1
triton==3.2.0
typing_extensions==4.12.2
tzdata==2024.1
urllib3==2.3.0
zipp==3.21.0
EOL

source ~/.bashrc
which python
python -m pip install -r /home/chetana/requirements.txt
# clean up
# rm requirements.txt
