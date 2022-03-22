import collections


ModelPath = collections.namedtuple('ModelPath', 'root_dir model_name')


def get_aid(model_name: str):
    _, aid, _ = model_name.split('/')
    aid = eval(aid[1:])
    assert isinstance(aid, int), aid
    return aid

def get_vid(model_name: str):
    _, _, vid = model_name.split('/')
    vid = eval(vid[1:])
    assert isinstance(vid, float), vid
    return vid

def get_aid_vid(model_name: str):
    _, aid, vid = model_name.split('/')
    aid = eval(aid[1:])
    vid = eval(vid[1:])
    assert isinstance(aid, int), aid
    assert isinstance(vid, float), vid
    return aid, vid
