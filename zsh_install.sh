set -e

cp ~/chenxinwei/.zshrc ~/.zshrc

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
yes y | sudo apt-get install mosh
# yes y | sudo apt-get install -y locales

yes y | sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip

# echo "nameserver 114.114.114.114" | sudo tee -a /etc/resolv.conf
# echo "nameserver 8.8.8.8" | sudo tee -a /etc/resolv.conf
# cd /etc/apt/sources.list.d
# sudo mv cuda.list cuda.list-old
# sudo mv nvidia-ml.list nvidia-ml.list-old
cd ~
yes y | sudo apt-get install libboost-all-dev
sudo chown -R ubuntu ~/.condarc
conda config --add envs_dirs $HOME/chenxinwei/conda/envs
# yes y | conda create -n grl python==3.8.10

# conda activate grl

# yes y | conda install cudnn
# pip install --upgrade pip

# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade setuptools psutil wheel
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gym atari_py procgen mujoco-py mujoco
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gfootball
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple jax optax haiku chex
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ray
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scipy pandas Pillow matplotlib plotly
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ipython
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tqdm
# pip install pysc2

# rsync -avz ~/chenxinwei/StarCraftII ~/
rsync -avz ~/chenxinwei/.mujoco ~/