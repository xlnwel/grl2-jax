import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1

from utility.utils import squarest_grid_size


def image_summary(name, images, step=None):
    if len(images.shape) == 3:
        images = images[None]
    if np.issubdtype(images.dtype, np.floating):
        images = np.clip(255 * images, 0, 255).astype(np.uint8)
    img = merge(images)
    tf.summary.image(name + '/grid', img, step)
    

def video_summary(name, video, step=None, fps=20):
    name = name if isinstance(name, str) else name.decode('utf-8')
    if np.issubdtype(video.dtype, np.floating):
        video = np.clip(255 * video, 0, 255).astype(np.uint8)
    B, T, H, W, C = video.shape
    try:
        frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
        summary = tf1.Summary()
        image = tf1.Summary.Image(height=B * H, width=T * W, colorspace=C)
        image.encoded_image_string = encode_gif(frames, fps)
        summary.value.add(tag=name + '/gif', image=image)
        tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
    except (IOError, OSError) as e:
        print('GIF summaries require ffmpeg in $PATH.', e)
        frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
        tf.summary.image(name + '/grid', frames, step)


def merge(images, size=None):
    assert images.shape.ndims == 4, f'images should be 4D, but get shape {images.shape}'
    if size is None:
        size = squarest_grid_size(images.shape[0])
    h, w = images.shape[1], images.shape[2]
    image_type = images.dtype
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c), dtype=image_type)
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        if np.issubdtype(image_type, np.uint8):
            return img
        if np.min(img) < -.5:
            # for images in range [-1, 1], make it in range [0, 1]
            img = (img + 1) / 2
        elif np.min(img) < 0:
            # for images in range [-.5, .5]
            img = img + .5
        img = np.clip(255 * img, 0, 255).astype(np.uint8)
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]), dtype=image_type)
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        NotImplementedError


def encode_gif(frames, fps):
    """ encode gif from frames in another process, return a gif """
    from subprocess import Popen, PIPE
    h, w, c = frames[0].shape
    pxfmt = {1: 'gray', 3: 'rgb24'}[c]
    cmd = ' '.join([
        f'ffmpeg -y -f rawvideo -vcodec rawvideo',
        f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
        f'[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
        f'-r {fps:.02f} -f gif -'])
    proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in frames:
        proc.stdin.write(image.tostring())
    out, err = proc.communicate()
    if proc.returncode:
        raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
    del proc
    return out
