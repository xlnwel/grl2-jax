import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm',
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
    parser.add_argument('--trials', '-t',
                        type=int,
                        default=1,
                        help='number of trials')
    parser.add_argument('--prefix', '-p',
                        default='',
                        help='directory prefix')
    parser.add_argument('--model-name', '-mn',
                        default='',
                        help='model name')
    parser.add_argument('--directory', '-d',
                        type=str,
                        default='',
                        help='directory where checkpoints and "config.yaml" exist')
    parser.add_argument('--logdir', '-ld',
                        type=str,
                        default='logs')
    parser.add_argument('--grid-search', '-gs',
                        action='store_true')
    parser.add_argument('--delay',
                        default=1,
                        type=int)
    parser.add_argument('--verbose', '-v',
                        type=str,
                        default='warning')
    args = parser.parse_args()

    return args