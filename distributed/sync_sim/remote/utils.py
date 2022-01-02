from utility.utils import eval_str


def get_aid_vid(model_name):
    _, aid, vid = model_name.split('/')
    aid = eval_str(aid.split('-')[-1])
    vid = eval_str(vid[1:])
    return aid, vid
