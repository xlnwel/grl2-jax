from core.mixin.monitor import create_recorder, create_tensorboard_writer


class Monitor:
    def __init__(self, root_dir, model_name, name):
        self._root_dir = root_dir
        self._model_name = model_name
        self._recorder = create_recorder(
            root_dir=root_dir, model_name=model_name)
        self._tb_writer = create_tensorboard_writer(
            root_dir=root_dir, model_name=model_name, name=name)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"Attempted to get missing private attribute '{name}'")
        if hasattr(self._recorder, name):
            return getattr(self._recorder, name)
        elif hasattr(self._tb_writer, name):
            return getattr(self._tb_writer, name)
        raise AttributeError(f"Attempted to get missing attribute '{name}'")

    def record(self, step):
        record(
            recorder=self._recorder, 
            tb_writer=self._tb_writer, 
            model_name=self._model_name, 
            step=step
        )


def record(*, recorder, tb_writer, model_name, 
           prefix=None, step, print_terminal_info=True, 
           **kwargs):
    stats = dict(
        model_name=f'{model_name}',
        steps=step,
        **recorder.get_stats(**kwargs),
    )
    if tb_writer is not None:
        tb_writer.scalar_summary(stats, prefix=prefix, step=step)
        tb_writer.flush()
    if recorder is not None:
        recorder.record_stats(stats, print_terminal_info=print_terminal_info)


def create_monitor(root_dir, model_name, name):
    return Monitor(root_dir, model_name, name)
