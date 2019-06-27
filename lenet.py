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
from data_utils import MNISTDataset
from tqdm import tqdm

batch_size = 64
epochs = 10
learning_rate = 1e-2
weight_decay = 0

train_dataset = MNISTDataset('./data/mnist', batch_size=batch_size, shuffle=True)
test_dataset = MNISTDataset('./data/mnist', batch_size=None, kind='t10k', shuffle=False)


class Lenet(Module):
    def __init__(self):
        super(Lenet, self).__init__()

        self.conv1 = Conv2D(1, 12, 5)
        self.relu1 = Relu()
        self.pool1 = MaxPooling(ksize=2)
        self.conv2 = Conv2D(12, 12, 3)
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
optimizer = SGD(model.parameters(), learning_rate, weight_decay=weight_decay)

for epoch in range(epochs):

    train_acc, val_acc = 0, 0
    train_loss, val_loss = 0, 0

    model.train()

    for images, labels in tqdm(train_dataset):

        optimizer.zero_grad()

        output = model(images)
        loss, grad = criterion(output, labels)

        train_acc += (np.argmax(output, axis=1) == labels).sum()

        model.backward(grad)
        optimizer.step(len(labels))

        train_loss += loss / len(labels)

    print("Time: {} Epoch: {} Train Acc: {:.2f} Train Loss: {:.4f}".
          format(time.strftime("%H:%M:%S"), epoch, train_acc * 100.0 / train_dataset.data_len, train_loss / len(train_dataset)))

    model.eval()
    # validation
    for images, labels in tqdm(test_dataset):
        output = model(images)

        loss, grad = criterion(output, labels)

        val_acc += (np.argmax(output, axis=1) == labels).sum()
        val_loss += loss / len(labels)

    print("Time: {} Epoch: {} Val Acc: {:.2f} Val Loss: {:.4f}".
          format(epoch, time.strftime("%H:%M:%S"), val_acc * 100.0 / test_dataset.data_len, val_loss / len(test_dataset)))

