import numpy as np
import triplen.nn as nn
import triplen.optim as optim
import time
from examples.mnist.data_utils import MNISTDataset
from tqdm import tqdm

np.random.seed(123)

batch_size = 64
epochs = 10
learning_rate = 1e-3
weight_decay = 0

train_dataset = MNISTDataset('./examples/mnist/data/mnist', batch_size=batch_size, shuffle=True)
test_dataset = MNISTDataset('./examples/mnist/data/mnist', batch_size=None, kind='t10k', shuffle=False)


class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()

        self.conv1 = nn.Conv2D(1, 12, 5)
        self.relu1 = nn.Relu()
        self.pool1 = nn.MaxPooling2D(kernel_size=2)
        self.conv2 = nn.Conv2D(12, 12, 3)
        self.relu2 = nn.Relu()
        self.pool2 = nn.MaxPooling2D(kernel_size=2)
        self.fc = nn.Linear(5 * 5 * 12, 10)

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

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)

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

