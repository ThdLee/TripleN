import numpy as np
import triplen
import triplen.nn as nn
import triplen.nn.functional as F
import triplen.optim as optim
import time
from examples.mnist.data_utils import MNISTDataset
from tqdm import tqdm

np.random.seed(123)

batch_size = 64
epochs = 10
learning_rate = 1e-3
weight_decay = 1e-5

train_dataset = MNISTDataset('./examples/mnist/data/mnist', batch_size=batch_size, shuffle=True)
test_dataset = MNISTDataset('./examples/mnist/data/mnist', batch_size=128, kind='t10k', shuffle=False)


class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()

        self.conv1 = nn.Conv2D(1, 20, 5)
        self.conv2 = nn.Conv2D(20, 50, 5)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = Lenet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)

for epoch in range(epochs):

    train_acc, val_acc = 0, 0
    train_loss, val_loss = 0, 0

    model.train()

    for images, labels in tqdm(train_dataset):
        images = triplen.Tensor(images)
        labels = triplen.Tensor(labels)

        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        train_acc += (np.argmax(output.data, axis=1) == labels.data).sum()

        loss.backward()
        optimizer.step(len(labels))

        train_loss += loss.item()

    print("Time: {} Epoch: {} Train Acc: {:.2f} Train Loss: {:.4f}".
          format(time.strftime("%H:%M:%S"), epoch, train_acc * 100.0 / train_dataset.data_len, train_loss / len(train_dataset)))

    model.eval()
    # validation
    for images, labels in tqdm(test_dataset):
        images = triplen.Tensor(images)
        labels = triplen.Tensor(labels)

        output = model(images)

        loss = criterion(output, labels)

        val_acc += (np.argmax(output.data, axis=1) == labels.data).sum()
        val_loss += loss.item()

    print("Time: {} Epoch: {} Val Acc: {:.2f} Val Loss: {:.4f}".
          format(epoch, time.strftime("%H:%M:%S"), val_acc * 100.0 / test_dataset.data_len, val_loss / len(test_dataset)))

