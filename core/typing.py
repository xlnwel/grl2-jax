import collections

"""
root_dir: logdir/env_name/algo_name
model_name: base_name/a{id}/i{iteration}-v{version}
"""
ModelPath = collections.namedtuple('ModelPath', 'root_dir model_name')


def construct_model_name_from_version(base, iteration, version):
    return f'{base}/i{iteration}-v{version}'

def construct_model_name(base, aid, iteration, version):
    return f'{base}/a{aid}/i{iteration}-v{version}'

def get_aid(model_name: str):
    _, aid, _ = model_name.rsplit('/', maxsplit=2)
    aid = eval(aid[1:])
    assert isinstance(aid, int), aid
    return aid

def get_vid(model_name: str):
    _, vid = model_name.rsplit('/', maxsplit=1)
    vid = vid.rsplit('v', maxsplit=1)[-1]
    return vid

def get_aid_vid(model_name: str):
    _, aid, vid = model_name.rsplit('/', maxsplit=2)
    aid = eval(aid[1:])
    vid = vid.rsplit('v', maxsplit=1)[-1]
    assert isinstance(aid, int), aid
    return aid, vid

def get_all_ids(model_name: str):
    _, aid, vid = model_name.rsplit('/', maxsplit=2)
    aid = eval(aid[1:])
    iid, vid = vid.split('v', maxsplit=1)
    iid = eval(iid[1:-1])
    assert isinstance(aid, int), aid
    assert isinstance(iid, int), iid
    return aid, iid, vid

def get_basic_model_name(model_name: str):
    """ Basic model name excludes aid and vid """
    name = '/'.join(model_name.split('/')[:2])

    return name

def get_algo(model: ModelPath):
    s = model.root_dir.split('/')
    if len(s) == 1:
        algo = model.root_dir
    elif len(s) == 3:
        algo = s[-1]
    elif len(s) == 4:
        algo = ''.join(s[-2:])
    else:
        # raise ValueError(f'Unknown model: {model}')
        assert False, f'Unknown model: {model}'

    return algo
