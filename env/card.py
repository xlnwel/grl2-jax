def info_func(agent, info):
    if isinstance(info, list):
        won = [i['won'] for i in info]
    else:
        won = info['won']
    agent.store(win_rate=won)
