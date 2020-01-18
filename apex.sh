source activate drl

python run/train.py -a apex-sac -e Humanoid-v2 -ms 2e7
# python run/train.py -a apex-sac -e Walker2d-v2 -ms 1.5e7
python run/train.py -a apex-sac -e Ant-v2 -ms 2e7
python run/train.py -a apex-sac -e Hopper-v2 -ms 2e7
python run/train.py -a apex-sac -e Reacher-v2 -ms 2e7
