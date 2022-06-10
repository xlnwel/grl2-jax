from core.mixin.monitor import Recorder, TensorboardWriter
from core.remote.base import RayBase
from core.typing import ModelPath


class Monitor(RayBase):
    def __init__(
        self, 
        model_path: ModelPath, 
        name='monitor', 
        use_recorder=True, 
        use_tensorboard=True
    ):
        self._model_path = model_path
        self._name = name
        self._use_recorder = use_recorder
        self._use_tensorboard = use_tensorboard
        self._step = None
        self._build(model_path)
    
    def _build(self, model_path):
        # we create a recorder anyway, but we do not store any data to the disk if use_recorder=False
        self._recorder = Recorder(self._model_path)
        if self._use_tensorboard and self._model_path is not None:
            self._tb_writer = TensorboardWriter(
                model_path=model_path, name=self._name)
        else:
            self._tb_writer = None

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"Attempted to get missing private attribute '{name}'")
        if self._recorder is not None and hasattr(self._recorder, name):
            return getattr(self._recorder, name)
        elif self._tb_writer is not None and hasattr(self._tb_writer, name):
            return getattr(self._tb_writer, name)
        raise AttributeError(f"Attempted to get missing attribute '{name}'")

    def reset_model_path(self, model_path: ModelPath):
        self._model_path = model_path
        self._build(model_path)

    def set_step(self, step):
        self._step = step

    def record(self, step=None, adaptive=True, print_terminal_info=True):
        record(
            recorder=self._recorder, 
            tb_writer=self._tb_writer, 
            model_name=self._model_path.model_name, 
            step=self._step if step is None else step,
            print_terminal_info=print_terminal_info, 
            adaptive=adaptive
        )


def record(
    *, 
    recorder: Recorder, 
    tb_writer: TensorboardWriter, 
    model_name: str, 
    prefix: str=None, 
    step: int, 
    print_terminal_info=True, 
    **kwargs
):
    stats = dict(
        model_name=f'{model_name}',
        steps=step,
        **recorder.get_stats(**kwargs)
    )
    if tb_writer is not None:
        tb_writer.scalar_summary(stats, prefix=prefix, step=step)
        tb_writer.flush()
    if recorder is not None:
        recorder.record_stats(stats, print_terminal_info=print_terminal_info)


def create_monitor(
    model_path: ModelPath, 
    name: str=None, 
    central_monitor: bool=False,
    use_recorder: bool=True, 
    use_tensorboard: bool=True
):
    if central_monitor:
        return Monitor.as_remote()(
            model_path, name, use_recorder, use_tensorboard)
    else:
        return Monitor(model_path, name, use_recorder, use_tensorboard)
