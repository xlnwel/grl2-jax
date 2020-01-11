source activate drl

python run/train.py -a asap2-sac -e Humanoid-v3
python run/train.py -a asap2-sac -e Walker2d-v3
python run/train.py -a asap2-sac -e Hopper-v3
python run/train.py -a asap2-sac -e Swimmer-v3
python run/train.py -a asap2-sac -e Ant-v3
# python run/train.py -a asap2-sac -e HalfCheetah-v3