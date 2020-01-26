source activate drl

python run/train.py -a asap2-sac -e Walker2d-v2
python run/train.py -a asap2-sac -e HalfCheetah-v2
# python run/train.py -a asap2-sac -e Swimmer-v2 -ms 2e7
# python run/train.py -a asap2-sac -e Reacher-v2
# python run/train.py -a asap2-sac -e Humanoid-v2
# python run/train.py -a asap2-sac -e Hopper-v2
# python run/train.py -a asap2-sac -e Ant-v2
