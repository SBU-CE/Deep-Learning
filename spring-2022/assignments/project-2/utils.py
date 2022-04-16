import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

import os
from six.moves.urllib.request import urlretrieve
import tarfile
import pickle
import sys
from urllib.error import *


def load_cifar10(transpose=False):
    """Loads CIFAR10 dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = 'cifar-10-batches-py'
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = get_file(dirname, origin=origin, untar=True)

    num_train_samples = 50000

    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')

    # 50000 training set
    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels

    # 10000 test set
    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if transpose:
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)
    return (x_train, y_train), (x_test, y_test)


def get_file(fname,
             origin,
             untar=False,
             cache_dir='data'):
    datadir = os.path.join(cache_dir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    print('File path: %s' % fpath)
    if not os.path.exists(fpath):
        print('Downloading data from', origin)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath)
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

    if untar:
        if not os.path.exists(untar_fpath):
            print('Extracting file.')
            with tarfile.open(fpath) as archive:
                archive.extractall(datadir)
        return untar_fpath

    return fpath


def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = pickle.load(f)
    else:
        d = pickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    f.close()
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def rgb2label(rgb_images, colours, batch_size):
    """
    Get colour categories given RGB values. This function doesn't
    actually do the work, instead it splits the work into smaller
    chunks that can fit into memory, and calls helper function
    _rgb2label

    Args:
      rgb_images: float numpy array of RGB images in [B, C, H, W] format
      colours: numpy array of colour categories and their RGB values
      batch_size: int value to determine size of batch
    Returns:
      result: int numpy array of shape [B, 1, H, W]
    """
    if np.shape(rgb_images)[0] < batch_size:
        return _rgb2label(rgb_images)
    nexts = []
    for i in range(0, np.shape(rgb_images)[0], batch_size):
        next = _rgb2label(rgb_images[i:i + batch_size, :, :, :], colours)
        nexts.append(next)
    result = np.concatenate(nexts, axis=0)
    return result


def _rgb2label(rgb_images, colours):
    """
    Get colour categories given RGB values. This is done by choosing
    the colour in `colours` that is the closest (in RGB space) to
    each point in the image `rgb_images`. This function is a little memory
    intensive, and so the size of `rgb_images` should not be too large.

    Args:
      rgb_images: float numpy array of RGB images in [B, C, H, W] format
      colours: numpy array of colour categories and their RGB values
    Returns:
      result: int numpy array of shape [B, 1, H, W]
    """
    num_colours = np.shape(colours)[0]
    rgb_images = np.expand_dims(rgb_images, 0)
    cs = np.reshape(colours, [num_colours, 1, 3, 1, 1])
    # Calculate distance
    dists = np.linalg.norm(rgb_images - cs, axis=2)  # 2 = colour axis
    label = np.argmin(dists, axis=0)
    label = np.expand_dims(label, axis=1)
    return label


def cat2rgb(labels, colours):
    """
    Get RGB colours given the colour categories

    Args:
      labels: integer numpy array of colour categories
      colours: numpy array of colour categories and their RGB values
    Returns:
      numpy tensor of RGB colours
    """
    return colours[labels]


def process(xs, ys, category_id, max_pixel=256.0, downsize_input=False):
    """
    Pre-process CIFAR10 images by taking only the car category,
    shuffling, and have colour values be bound between 0 and 1

    Args:
      xs: the colour RGB pixel values
      ys: the category labels
      category_id: int value as the id of each category in CIFAR10 dataset
      max_pixel: maximum pixel value in the original data
      downsize_input: boolean value
    Returns:
      xs: value normalized and shuffled colour images
      grey: greyscale images, also normalized so values are between 0 and 1
    """
    xs = xs / max_pixel
    xs = xs[np.where(ys == category_id)[0], :, :, :]
    np.random.shuffle(xs)

    grey = np.mean(xs, axis=1, keepdims=True)

    if downsize_input:
        downsize_module = nn.Sequential(nn.AvgPool2d(2),
                                        nn.AvgPool2d(2),
                                        nn.Upsample(scale_factor=2),
                                        nn.Upsample(scale_factor=2))
        xs_downsized = downsize_module.forward(torch.from_numpy(xs).float())
        xs_downsized = xs_downsized.data.numpy()
        return xs, xs_downsized
    else:
        return xs, grey


def plot(input, gtlabel, output, colours, path, visualize, compare_bilinear=False):
    """
    Generate png plots of input, ground truth, and outputs (10 samples you can change this value)

    Args:
      input: the greyscale input to the colourization CNN
      gtlabel: the grouth truth categories for each pixel
      output: the predicted categories for each pixel
      colours: numpy array of colour categories and their RGB values
      path: output path
      visualize: display the figures inline or save the figures in path
    """
    grey = np.transpose(input[:10, :, :, :], [0, 2, 3, 1])
    gtcolor = cat2rgb(gtlabel[:10, 0, :, :], colours)
    predcolor = cat2rgb(output[:10, 0, :, :], colours)

    img_stack = [
        np.hstack(np.tile(grey, [1, 1, 1, 3])),
        np.hstack(gtcolor),
        np.hstack(predcolor)]

    if compare_bilinear:
        downsize_module = nn.Sequential(nn.AvgPool2d(2),
                                        nn.AvgPool2d(2),
                                        nn.Upsample(scale_factor=2, mode='bilinear'),
                                        nn.Upsample(scale_factor=2, mode='bilinear'))
        gt_input = np.transpose(gtcolor, [0, 3, 1, 2, ])
        color_bilinear = downsize_module.forward(torch.from_numpy(gt_input).float())
        color_bilinear = np.transpose(color_bilinear.data.numpy(), [0, 2, 3, 1])
        img_stack = [
            np.hstack(np.transpose(input[:10, :, :, :], [0, 2, 3, 1])),
            np.hstack(gtcolor),
            np.hstack(predcolor),
            np.hstack(color_bilinear)]
    img = np.vstack(img_stack)

    plt.grid('off')
    plt.imshow(img, vmin=0., vmax=1.)
    if visualize:
        plt.show()
    else:
        plt.savefig(path)
