from core.elements.builder import ElementsBuilder
from core.utils import save_config
from env.func import get_env_stats
from utility.timer import Every, Timer


def main(config):
    env_stats = get_env_stats(config.env)
    builder = ElementsBuilder(config, env_stats, name='gd_zero')
    elements = builder.build_agent_from_scratch()
    agent = elements.agent

    tt = Timer('train')
    lt = Timer('log')
    to_record = Every(agent.LOG_PERIOD, agent.LOG_PERIOD)
    def record_stats(step):
        with lt:
            agent.store(**{
                'misc/train_step': agent.get_train_step(),
                'time/train': tt.total(),
                'time/log': lt.total(),
                'time/train_mean': tt.average(),
                'time/log_mean': lt.average(),
            })
            agent.record(step=step)
            agent.save()

    step = 0
    while True:
        start_train_step = step
        with tt:
            agent.train_record()
        step = agent.get_train_step()
        agent.store(
            tps=(step-start_train_step)/tt.last())
        agent.set_env_step(step)

        if to_record(step):
            record_stats(step)