source activate drl

python run/train.py -a apex-sac -e HalfCheetah-v2 -p 3
python run/train.py -a apex-sac -e Walker2d-v2 -p 2
python run/train.py -a apex-sac -e Walker2d-v2 -p 3
python run/train.py -a apex-sac -e Hopper-v2 -p 3
# python run/train.py -a apex-sac -e Swimmer-v2 -ms 2e7
# python run/train.py -a apex-sac -e Reacher-v2
# python run/train.py -a apex-sac -e Humanoid-v2 -ms 3e7 -p n_envs=2
# python run/train.py -a apex-sac -e Ant-v2 -ms 2e7
# python run/train.py -a apex-sac -e Hopper-v2 -ms 2e7
