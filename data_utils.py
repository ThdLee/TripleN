import numpy as np
import struct
import math
from glob import glob


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    images_path = glob('./%s/%s*3-ubyte' % (path, kind))[0]
    labels_path = glob('./%s/%s*1-ubyte' % (path, kind))[0]

    with open(labels_path, 'rb') as lbpath:
        struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 28, 28, 1)

    return images, labels


class MNISTDataset(object):
    def __init__(self, path, batch_size=32, kind='train', shuffle=True):
        images, labels = load_mnist(path, kind=kind)
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_len = len(self.labels)

    def __len__(self):
        if self.batch_size is None:
            return 1
        else:
            return math.ceil(len(self.images) / self.batch_size)

    def __iter__(self):
        if self.shuffle:
            index = list(range(len(self.labels)))
            np.random.shuffle(index)
            self.images = self.images[index]
            self.labels = self.labels[index]
        if self.batch_size is None:
            yield self.images, self.labels
        else:
            start, end = 0, self.batch_size
            while start < len(self.labels):
                if end > len(self.labels):
                    images = self.images[start:]
                    labels = self.labels[start:]
                else:
                    images = self.images[start:end]
                    labels = self.labels[start:end]

                yield images, labels

                start += self.batch_size
                end += self.batch_size
