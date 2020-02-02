import argparse


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', '-a',
                        type=str,
                        nargs='*')
    parser.add_argument('--environment', '-e',
                        type=str,
                        nargs='*',
                        default=[''])
    parser.add_argument('--kwargs', '-kw',
                        type=str,
                        nargs='*',
                        default=[])
    parser.add_argument('--render', '-r',
                        action='store_true',
                        help='render the environment. this currently does not work for EnvVec')
    parser.add_argument('--trials', '-t',
                        type=int,
                        default=1,
                        help='number of trials')
    parser.add_argument('--prefix', '-p',
                        default='',
                        help='prefix for model dir')
    parser.add_argument('--directory', '-d',
                        type=str,
                        default='',
                        help='directory where checkpoints and "config.yaml" exist')
    parser.add_argument('--grid_search', '-gs',
                        action='store_true')
    parser.add_argument('--max_steps', '-ms',
                        default=None,
                        type=float)
    parser.add_argument('--n_envs', '-ne',
                        default=None,
                        type=int)
    parser.add_argument('--delay',
                        default=1,
                        type=int)
    args = parser.parse_args()

    return args