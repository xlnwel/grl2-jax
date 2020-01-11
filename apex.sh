source activate drl

python run/train.py -a apex-sac -e Humanoid-v3
python run/train.py -a apex-sac -e Walker2d-v3
python run/train.py -a apex-sac -e Hopper-v3
python run/train.py -a apex-sac -e Swimmer-v3
python run/train.py -a apex-sac -e Ant-v3
python run/train.py -a apex-sac -e HalfCheetah-v3