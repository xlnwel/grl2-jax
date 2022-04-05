# yes y | sudo apt-get update
# yes y | sudo apt-get install tmux
# yes y | sudo apt-get install nano vim
# yes y | sudo apt-get install rsync
# sh ~/.oh-my-zsh/install-oh-my-zsh.sh
# chsh -s `which zsh`
# rsync -avz ~/chenxinwei/.zshrc ~/
# rsync -avz ~/chenxinwei/.script ~/
# source .zshrc

# setup mosh
# yes y | sudo apt-get update
# yes y | sudo apt-get install mosh
# yes y | sudo apt-get install -y locales
# # install en_US.UTF-8
# sudo locale-gen en_US.UTF-8
# # setup locales in ~/.bashrc or ~/.zshrc
# export LANG=en_US.UTF-8
# export LC_ALL=en_US.UTF-8

yes y | sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip

# echo "nameserver 114.114.114.114" >> sudo /etc/resolv.conf
# echo "nameserver 8.8.8.8" >> sudo /etc/resolv.conf
# yes y | sudo apt-get install libboost-all-dev
# yes y | conda create -n grl python==3.8.10

# source activate grl

# yes y | conda install cudnn
# pip install --upgrade pip

# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade setuptools psutil wheel
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gym atari_py
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gfootball
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow-gpu tensorflow-probability
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ray
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scipy pandas Pillow plotly opencv-python
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ipython
