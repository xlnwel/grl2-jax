import io
import os
import math
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from matplotlib import colors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utility.utils import squarest_grid_size


def construct_df(matrix, col_name='y', row_name='x', col_labels=None, row_labels=None):
    col = np.arange(matrix.shape[0])
    row = np.arange(matrix.shape[1])
    d = {
        col_name: np.tile(col.reshape(-1, 1), (1, matrix.shape[1])).reshape(-1),
        row_name: np.tile(row, matrix.shape[0]),
        'val': matrix.reshape(-1)
    }
    df = pd.DataFrame(data=d)
    if col_labels:
        df[col_name].replace(col, col_labels, inplace=True)
    if row_labels:
        df[row_name].replace(row, row_labels, inplace=True)
    df = df.pivot(col_name, row_name, 'val')
    return df


def decode_png(png):
    image = tf.image.decode_png(png, channels=4)
    return image


def get_tick_labels(size, threshould=10):
    return math.ceil(size / threshould) if size > threshould else 'auto'


def matrix_plot(
    matrix, 
    figsize=6, 
    label_top=False, 
    label_bottom=True, 
    anno_max=10, 
    save_path=None, 
    xlabel=None, 
    ylabel=None, 
    xticklabels='auto', 
    yticklabels='auto', 
    xticklabelnames=None,
    yticklabelnames=None,
    dpi=300
):
    fig = plt.figure(0, figsize=(figsize, 4/5 * figsize), dpi=dpi)
    ax = fig.add_subplot(111)
    def get_norm(matrix):        
        vmin = np.min(matrix)
        vmax = np.max(matrix)
        if vmax - vmin > 1000 and vmin >= 0:
            new_matrix = matrix.astype(np.float32)
            new_matrix[new_matrix==0] = np.nan
            vmin = np.nanmin(new_matrix)
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            if vmin < 0 and vmax > 0:
                norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            else:
                norm = colors.Normalize(vmin=vmin, vmax=vmax)

        return norm

    def get_kwargs(matrix):
        kwargs = {
            'cmap': 'YlGnBu', 
            'norm': get_norm(matrix), 
            'center': 0, 
            'xticklabels': xticklabels, 
            'yticklabels': yticklabels, 
            'ax': ax
        }
        if matrix.shape[0] < anno_max and matrix.shape[1] < anno_max:
            v = matrix.max()
            kwargs.update({
                'annot': True, 
                'fmt': f'.2f' if v < 0 else f'.2g'
            })
        return kwargs

    kwargs = get_kwargs(matrix)
    pd = construct_df(
        matrix, 
        col_name=ylabel, 
        row_name=xlabel, 
        col_labels=yticklabelnames, 
        row_labels=xticklabelnames, 
    )
    ax = sns.heatmap(pd, **kwargs)
    ax.tick_params(
        axis='both',        # changes apply to the x-axis
        which='both',       # both major and minor ticks are affected
        labeltop=label_top, 
        labelbottom=label_bottom, 
        left=False, 
        bottom=False
    )

    ax.xaxis.set_label_position('top')

    if save_path:
        plt.savefig(f'{save_path}.pdf')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi)
    plt.close()
    buf.seek(0)
    image = decode_png(buf.getvalue())
    buf.close()

    return image

def _matrix_plot_test():
    m = 5
    n = 10
    payoff = np.logspace(0, 100, m*n).reshape((m, n)) / m / n

    buf = matrix_plot(
        payoff, 
        label_top=True, 
        label_bottom=False, 
        xlabel='Opponents', 
        ylabel='Time', 
        xticklabels=get_tick_labels(payoff.shape[1]), 
        yticklabels=get_tick_labels(payoff.shape[0]),
        yticklabelnames=list('abcde'), 
    )
    image = decode_png(buf.getvalue())
    plt.imshow(image)

    payoff = np.arange(m*n).reshape((m, n)) / m / n - .3
    print(payoff)
    matrix_plot(
        payoff, 
        label_top=True, 
        label_bottom=False, 
        xlabel='Opponents', 
        ylabel='Time', 
        xticklabels=get_tick_labels(payoff.shape[1]), 
        yticklabels=get_tick_labels(payoff.shape[0]),
        yticklabelnames=list('abcde'), 
    )
    image = decode_png(buf.getvalue())
    plt.imshow(image)

    payoff = np.arange(n*n).reshape((n, n)) / n / n - .4
    matrix_plot(
        payoff, figsize=6, label_top=True, 
        label_bottom=False, invert_yaxis=True
    )
    plt.savefig('linear_matrix.pdf')


def grid_placed(images, size=None):
    assert len(images.shape) == 4, f'images should be 4D, but get shape {images.shape}'
    B, H, W, C = images.shape
    if size is None:
        size = squarest_grid_size(B)
    image_type = images.dtype
    if (images.shape[3] in (3,4)):
        img = np.zeros((H * size[0], W * size[1], C), dtype=image_type)
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * H:j * H + H, i * W:i * W + W, :] = image
        if np.issubdtype(image_type, np.uint8):
            return img
        if np.min(img) < -.5:
            # for images in range [-1, 1], make it in range [0, 1]
            img = (img + 1) / 2
        elif np.min(img) < 0:
            # for images in range [-.5, .5]
            img = img + .5
        assert np.min(img) >= 0, np.min(img)
        assert np.max(img) <= 1, np.max(img)
        img = np.clip(255 * img, 0, 255).astype(np.uint8)
        return img
    elif images.shape[3]==1:
        img = np.zeros((H * size[0], W * size[1]), dtype=image_type)
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * H:j * H + H, i * W:i * W + W] = image[:,:,0]
        return img
    else:
        NotImplementedError


def encode_gif(frames, fps):
    """ encode gif from frames in another process, return a gif """
    from subprocess import Popen, PIPE
    H, W, C = frames[0].shape
    pxfmt = {1: 'gray', 3: 'rgb24'}[C]
    cmd = ' '.join([
        f'ffmpeg -y -f rawvideo -vcodec rawvideo',
        f'-r {fps:.02f} -s {W}x{H} -pix_fmt {pxfmt} -i - -filter_complex',
        f'[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
        f'-r {fps:.02f} -f gif -'])
    proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in frames:
        proc.stdin.write(image.tobytes())
    out, err = proc.communicate()
    if proc.returncode:
        raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
    del proc
    return out


def save_video(name, video, fps=30, out_dir='results'):
    name = name if isinstance(name, str) else name.decode('utf-8')
    video = np.array(video, copy=False)
    if np.issubdtype(video.dtype, np.floating):
        video = np.clip(255 * video, 0, 255).astype(np.uint8)
    while len(video.shape) < 5:
        video = np.expand_dims(video, 0)
    B, T, H, W, C = video.shape
    if B != 1:
        # merge multiple videos into a single video
        bh, bw = squarest_grid_size(B)
        frames = video.reshape((bh, bw, T, H, W, C))
        frames = frames.transpose((2, 0, 3, 1, 4, 5))
        frames = frames.reshape((T, bh*H, bw*W, C))
    else:
        frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, W, C))
    f1, *frames = [Image.fromarray(f) for f in frames]
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    path = f'{out_dir}/{name}.gif'
    f1.save(fp=path, format='GIF', append_images=frames,
         save_all=True, duration=1000//fps, loop=0)
    print(f"video is saved to '{path}'")


""" summaries useful for core.log.graph_summary"""
def image_summary(name, images, step=None):
    # when wrapped by tf.numpy_function in @tf.function, str are 
    # represented by bytes so we need to convert it back to str
    name = name if isinstance(name, str) else name.decode('utf-8')
    if len(images.shape) == 3:
        images = images[None]
    if np.issubdtype(images.dtype, np.floating):
        assert np.logical_and(images >= 0, images <= 1).all()
        images = np.clip(255 * images, 0, 255).astype(np.uint8)
    img = np.expand_dims(grid_placed(images), 0)
    tf.summary.image(name + '/image', img, step)


def video_summary(name, video, size=None, fps=30, step=None):
    name = name if isinstance(name, str) else name.decode('utf-8')
    if np.issubdtype(video.dtype, np.floating):
        video = np.clip(255 * video, 0, 255).astype(np.uint8)
    while len(video.shape) < 5:
        video = np.expand_dims(video, 0)
    B, T, H, W, C = video.shape
    if size is None and B != 1:
        bh, bw = squarest_grid_size(B)
        frames = video.reshape((bh, bw, T, H, W, C))
        frames = frames.transpose((2, 0, 3, 1, 4, 5))
        frames = frames.reshape((T, bh*H, bw*W, C))
    else:
        if size is None:
            size = (1, 1)
        assert size[0] * size[1] == B, f'Size({size}) does not match the batch dimension({B})'
        frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, size[0]*H, size[1]*W, C))
    try:
        summary = tf1.Summary()
        image = tf1.Summary.Image(height=B * H, width=T * W, colorspace=C)
        image.encoded_image_string = encode_gif(frames, fps)
        summary.value.add(tag=name, image=image)
        tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
    except (IOError, OSError) as e:
        print('GIF summaries require ffmpeg in $PATH.', e)
        frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
        tf.summary.image(name + '/image', frames, step)
