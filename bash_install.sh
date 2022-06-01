# set -e

yes y | sudo apt-get update
yes y | sudo apt-get install tmux
yes y | sudo apt-get install nano vim
yes y | sudo apt-get install rsync
# setup zsh
yes y | sudo apt-get install zsh
rsync -avz ~/chenxinwei/.oh-my-zsh ~/
ls ~/.oh-my-zsh
chmod +x ~/.oh-my-zsh/install-oh-my-zsh.sh
zsh ~/.oh-my-zsh/install-oh-my-zsh.sh
chsh -s `which zsh`
rsync -avz ~/chenxinwei/.zshrc ~/
rsync -avz ~/chenxinwei/.script ~/
zsh
