source activate drl

python run/train.py -a asap-sac -e Walker2d-v2 -p temp01
python run/train.py -a asap-sac -e Hopper-v2 -p temp01
python run/train.py -a asap-sac -e Swimmer-v2 -p temp01
python run/train.py -a asap-sac -e Ant-v2 -p temp01
python run/train.py -a asap-sac -e HalfCheetah-v2 -p temp01
python run/train.py -a asap-sac -e Humanoid-v2 -p temp01
