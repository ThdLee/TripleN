import numpy as np
from nn.conv import Conv2D
from nn.linear import Linear
from nn.pooling import MaxPooling, AvgPooling
from nn.loss import CrossEntropyLoss
from nn.activation import Relu
from nn.module import Module
from optim.sgd import SGD
from optim.adam import Adam
import time
import struct
from glob import glob


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    images_path = glob('./%s/%s*3-ubyte' % (path, kind))[0]
    labels_path = glob('./%s/%s*1-ubyte' % (path, kind))[0]

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


images, labels = load_mnist('./data/mnist')
test_images, test_labels = load_mnist('./data/mnist', 't10k')

batch_size = 64


class Lenet(Module):
    def __init__(self):
        super(Lenet, self).__init__()

        self.conv1 = Conv2D(shape=(28, 28, 1), output_channels=12, ksize=5)
        self.relu1 = Relu()
        self.pool1 = MaxPooling(ksize=2)
        self.conv2 = Conv2D(shape=(12, 12, 12), output_channels=12, ksize=3)
        self.relu2 = Relu()
        self.pool2 = MaxPooling(ksize=2)
        self.fc = Linear(5 * 5 * 12, 10)

    def forward(self, input):
        conv1_out = self.pool1(self.relu1(self.conv1(input)))
        conv2_out = self.pool2(self.relu2(self.conv2(conv1_out)))
        output = self.fc(conv2_out)
        return output

    def backward(self, grad_output):
        self.conv1.backward(self.relu1.backward(self.pool1.backward(
            self.conv2.backward(self.relu2.backward(self.pool2.backward(
                self.fc.backward(grad_output)))))))


model = Lenet()

criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters())


# train_loss_record = []
# train_acc_record = []
# val_loss_record = []
# val_acc_record = []

for epoch in range(20):
    # if epoch < 5:
    #     learning_rate = 0.00001
    # elif epoch < 10:
    #     learning_rate = 0.000001
    # else:
    #     learning_rate = 0.0000001

    learning_rate = 1e-4

    batch_loss = 0
    batch_acc = 0
    val_acc = 0
    val_loss = 0

    # train
    train_acc = 0
    train_loss = 0
    for i in range(images.shape[0] // batch_size):
        img = images[i * batch_size:(i + 1) * batch_size].reshape([batch_size, 28, 28, 1])
        label = labels[i * batch_size:(i + 1) * batch_size]

        output = model(img)
        loss, grad = criterion(output, label)

        for j in range(batch_size):
            if np.argmax(output[j]) == label[j]:
                batch_acc += 1
                train_acc += 1

        model.backward(grad)
        optimizer.step()
        model.zero_grad()

        if i % 50 == 0:
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + \
                  "  epoch: %d ,  batch: %5d , avg_batch_acc: %.4f  avg_batch_loss: %.4f  learning_rate %f" % (epoch,
                                                                                                               i,
                                                                                                               batch_acc / float(
                                                                                                                   batch_size),
                                                                                                               loss / batch_size,
                                                                                                               learning_rate))

        batch_loss = 0
        batch_acc = 0

    print(time.strftime("%Y-%m-%d %H:%M:%S",
                            time.localtime()) + "  epoch: %5d , train_acc: %.4f  avg_train_loss: %.4f" % (
            epoch, train_acc / float(images.shape[0]), train_loss / images.shape[0]))

    # validation
    for i in range(test_images.shape[0] // batch_size):
        img = test_images[i * batch_size:(i + 1) * batch_size].reshape([batch_size, 28, 28, 1])
        label = test_labels[i * batch_size:(i + 1) * batch_size]
        output = model(img)

        for j in range(batch_size):
            if np.argmax(output[j]) == label[j]:
                val_acc += 1

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "  epoch: %5d , val_acc: %.4f  avg_val_loss: %.4f" % (
        epoch, val_acc / float(test_images.shape[0]), val_loss / test_images.shape[0]))

