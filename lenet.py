import numpy as np
from nn.conv import Conv2D
from nn.linear import Linear
from nn.pooling import MaxPooling, AvgPooling
from nn.loss import CrossEntropyLoss
from nn.activation import Relu
from nn.module import Module
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
        self.fc = Linear(10 * 10 * 12, 10)

    def forward(self, input):
        conv1_out = self.pool1(self.relu1(self.conv1(input)))
        conv2_out = self.pool2(self.relu2(self.conv2(conv1_out)))
        output = self.fc(conv2_out)
        return output

    def backward(self, grad):
        self.conv1.backward(self.relu1.backward(self.pool1.backward(
            self.conv2.backward(self.relu2.backward(self.pool2.backward(
                self.fc.backward(grad)))))))




conv1 = Conv2D([batch_size, 28, 28, 1], 12, 5, 1)
relu1 = Relu(conv1.output_shape)
pool1 = MaxPooling(relu1.output_shape)
conv2 = Conv2D(pool1.output_shape, 24, 3, 1)
relu2 = Relu(conv2.output_shape)
pool2 = MaxPooling(relu2.output_shape)
fc = Linear(pool2.output_shape, 10)
sf = CrossEntropyLoss(fc.output_shape)


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
        conv1_out = relu1.forward(conv1.forward(img))
        pool1_out = pool1.forward(conv1_out)
        conv2_out = relu2.forward(conv2.forward(pool1_out))
        pool2_out = pool2.forward(conv2_out)
        fc_out = fc.forward(pool2_out)
        batch_loss += sf.cal_loss(fc_out, np.array(label))
        train_loss += sf.cal_loss(fc_out, np.array(label))

        for j in range(batch_size):
            if np.argmax(sf.softmax[j]) == label[j]:
                batch_acc += 1
                train_acc += 1

        sf.gradient()
        conv1.bacward(relu1.gradient(pool1.gradient(
            conv2.bacward(relu2.gradient(pool2.gradient(
                fc.gradient(sf.eta)))))))

        if i % 1 == 0:
            fc.backward(alpha=learning_rate, weight_decay=0.0004)
            conv2.backward(alpha=learning_rate, weight_decay=0.0004)
            conv1.backward(alpha=learning_rate, weight_decay=0.0004)

            if i % 50 == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + \
                      "  epoch: %d ,  batch: %5d , avg_batch_acc: %.4f  avg_batch_loss: %.4f  learning_rate %f" % (epoch,
                                                                                                 i, batch_acc / float(
                          batch_size), batch_loss / batch_size, learning_rate))


            batch_loss = 0
            batch_acc = 0


    print(time.strftime("%Y-%m-%d %H:%M:%S",
                            time.localtime()) + "  epoch: %5d , train_acc: %.4f  avg_train_loss: %.4f" % (
            epoch, train_acc / float(images.shape[0]), train_loss / images.shape[0]))

    # validation
    for i in range(test_images.shape[0] / batch_size):
        img = test_images[i * batch_size:(i + 1) * batch_size].reshape([batch_size, 28, 28, 1])
        label = test_labels[i * batch_size:(i + 1) * batch_size]
        conv1_out = relu1.forward(conv1.forward(img))
        pool1_out = pool1.forward(conv1_out)
        conv2_out = relu2.forward(conv2.forward(pool1_out))
        pool2_out = pool2.forward(conv2_out)
        fc_out = fc.forward(pool2_out)
        val_loss += sf.cal_loss(fc_out, np.array(label))

        for j in range(batch_size):
            if np.argmax(sf.softmax[j]) == label[j]:
                val_acc += 1

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "  epoch: %5d , val_acc: %.4f  avg_val_loss: %.4f" % (
        epoch, val_acc / float(test_images.shape[0]), val_loss / test_images.shape[0]))

