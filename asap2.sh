source activate drl

python run/train.py -a asap2-sac -e Walker2d-v2 -ms 2e7
python run/train.py -a asap2-sac -e Humanoid-v2 -ms 2e7
# python run/train.py -a asap2-sac -e Hopper-v2 --fifo -ms 1e7
# python run/train.py -a asap2-sac -e Swimmer-v2 --fifo
# python run/train.py -a asap2-sac -e Ant-v2 --fifo -ms 2e7