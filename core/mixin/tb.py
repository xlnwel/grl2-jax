from typing import Union
import numpy as np
import tensorflow as tf

from core.typing import ModelPath
from tools import graph


""" Tensorboard Writer """
class TensorboardWriter:
    def __init__(self, model_path: ModelPath, name):
        self.model_path = model_path
        self._writer = create_tb_writer(model_path)
        self.name = name
        tf.summary.experimental.set_step(0)
    
    def set_summary_step(self, step):
        """ Sets tensorboard step """
        set_summary_step(step)

    def scalar_summary(self, stats, prefix=None, step=None):
        """ Adds scalar summary to tensorboard """
        scalar_summary(self._writer, stats, prefix=prefix, step=step)

    def histogram_summary(self, stats, prefix=None, step=None):
        """ Adds histogram summary to tensorboard """
        histogram_summary(self._writer, stats, prefix=prefix, step=step)

    def image_summary(self, images, name, prefix=None, step=None):
        image_summary(self._writer, images, name, prefix=prefix, step=step)

    def graph_summary(self, sum_type, *args, step=None):
        """ Adds graph summary to tensorboard
        This should only be called inside @tf.function
        Args:
            sum_type str: either "video" or "image"
            args: Args passed to summary function defined in utility.graph,
                of which the first must be a str to specify the tag in Tensorboard
        """
        assert isinstance(args[0], str), f'args[0] is expected to be a name string, but got "{args[0]}"'
        args = list(args)
        args[0] = f'{self.name}/{sum_type}/{args[0]}'
        graph_summary(self._writer, sum_type, args, step=step)

    def video_summary(self, video, step=None, fps=30):
        graph.video_summary(f'{self.name}/sim', video, fps=fps, step=step)

    def matrix_summary(
        self, 
        *, 
        model: ModelPath=None, 
        matrix, 
        label_top=True, 
        label_bottom=False, 
        xlabel, 
        ylabel, 
        xticklabels, 
        yticklabels,
        name, 
        step=None, 
    ):
        if model is None:
            model = self.model_path
        matrix_summary(
            model=model, 
            matrix=matrix, 
            label_top=label_top, 
            label_bottom=label_bottom, 
            xlabel=xlabel, 
            ylabel=ylabel, 
            xticklabels=xticklabels, 
            yticklabels=yticklabels,
            name=name, 
            writer=self._writer, 
            step=step, 
        )

    def flush(self):
        self._writer.flush()


""" Tensorboard Ops """
def set_summary_step(step):
    tf.summary.experimental.set_step(step)

def scalar_summary(writer, stats, prefix=None, step=None):
    if step is not None:
        tf.summary.experimental.set_step(step)
    prefix = prefix or 'stats'
    with writer.as_default():
        for k, v in stats.items():
            if isinstance(v, str):
                continue
            if '/' not in k:
                k = f'{prefix}/{k}'
            # print(k, np.array(v).dtype)
            tf.summary.scalar(k, tf.reduce_mean(v), step=step)

def histogram_summary(writer, stats, prefix=None, step=None):
    if step is not None:
        tf.summary.experimental.set_step(step)
    prefix = prefix or 'stats'
    with writer.as_default():
        for k, v in stats.items():
            if isinstance(v, (str, int, float)):
                continue
            if '/' not in k:
                k = f'{prefix}/{k}'
            tf.summary.histogram(k, v, step=step)

def graph_summary(writer, sum_type, args, step=None):
    """ This function should only be called inside a tf.function """
    fn = {'image': graph.image_summary, 'video': graph.video_summary}[sum_type]
    if step is None:
        step = tf.summary.experimental.get_step()
    def inner(*args):
        tf.summary.experimental.set_step(step)
        with writer.as_default():
            fn(*args)
    return tf.numpy_function(inner, args, [])

def image_summary(writer, images, name, prefix=None, step=None):
    if step is not None:
        tf.summary.experimental.set_step(step)
    if len(images.shape) == 3:
        images = images[None]
    if prefix:
        name = f'{prefix}/{name}'
    with writer.as_default():
        tf.summary.image(name, images, step=step)

def matrix_summary(
    *, 
    model: ModelPath, 
    matrix: np.ndarray, 
    label_top=True, 
    label_bottom=False, 
    xlabel: str, 
    ylabel: str, 
    xticklabels: Union[str, int, np.ndarray], 
    yticklabels: Union[str, int, np.ndarray],
    name, 
    writer, 
    step=None, 
):
    save_path = None if model is None else '/'.join([*model, name])
    image = graph.matrix_plot(
        matrix, 
        label_top=label_top, 
        label_bottom=label_bottom, 
        save_path=save_path, 
        xlabel=xlabel, 
        ylabel=ylabel, 
        xticklabels=xticklabels, 
        yticklabels=yticklabels
    )
    image_summary(writer, image, name, step=step)

def create_tb_writer(model_path: ModelPath):
    # writer for tensorboard summary
    # stats are saved in directory f'{root_dir}/{model_name}'
    writer = tf.summary.create_file_writer('/'.join(model_path))
    writer.set_as_default()
    return writer

def create_tensorboard_writer(model_path: ModelPath, name):
    return TensorboardWriter(model_path, name)
