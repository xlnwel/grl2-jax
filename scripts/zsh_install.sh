# set -e

# cp ~/chenxinwei/.zshrc ~/.zshrc

# apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
# sudo apt-key del 7fa2af80
# sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F60F4B3D7FA2AF80

# setup mosh
# yes y | sudo apt-get update
# yes y | sudo apt-get install -y locales
# # install en_US.UTF-8
# sudo locale-gen en_US.UTF-8
# # setup locales in ~/.bashrc or ~/.zshrc
# export LANG=en_US.UTF-8
# export LC_ALL=en_US.UTF-8
# # yes y | sudo apt-get install mosh
# yes y | sudo apt-get install -y locales

yes y | sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-python -m pip

# # echo "nameserver 114.114.114.114" | sudo tee -a /etc/resolv.conf
# # echo "nameserver 8.8.8.8" | sudo tee -a /etc/resolv.conf
# # cd /etc/apt/sources.list.d
# # sudo mv cuda.list cuda.list-old
# # sudo mv nvidia-ml.list nvidia-ml.list-old
# # cd ~
# yes y | sudo apt-get install libboost-all-dev
# cp -r ~/chenxinwei/.conda/envs/dreamer ~/.conda/envs
# # sudo chown -R ubuntu ~/.condarc
# # mkdir -p $HOME/.conda/envs
# # conda config --add envs_dirs $HOME/.conda/envs  # add $HOME/.conda/envs to conda environment directories
# yes y | conda create -n grl python==3.9.15
conda activate grl

# yes y | conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
# yes y | conda install -c conda-forge cudatoolkit=11.8.0
# python -m pip install nvidia-cudnn-cu11==8.6.0.163 "tensorflow==2.12.*"
# mkdir -p $CONDA_PREFIX/etc/conda/activate.d
# echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
python -m pip install --upgrade pip

python -m pip install --upgrade setuptools psutil wheel
python -m pip install opencv-python
python -m pip install gym==0.23.1
# python -m pip install atari_py procgen mujoco-py mujoco
python -m pip install jax optax dm-haiku distrax chex
python -m pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python -m pip install tensorflow_probability
python -m pip install ray
python -m pip install scipy pandas Pillow matplotlib plotly seaborn
python -m pip install ipython
python -m pip install tqdm
python -m pip install gfootball
# python -m pip install pysc2

# rsync -avz ~/chenxinwei/StarCraftII ~/
# rsync -avz ~/chenxinwei/.mujoco ~/
