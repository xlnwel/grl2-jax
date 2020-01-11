source activate drl

python run/train.py -a asap-sac -e Humanoid-v3
python run/train.py -a asap-sac -e Walker2d-v3
python run/train.py -a asap-sac -e Hopper-v3
python run/train.py -a asap-sac -e Swimmer-v3
python run/train.py -a asap-sac -e Ant-v3
python run/train.py -a asap-sac -e HalfCheetah-v3