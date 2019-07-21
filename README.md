# TripleN

TripleN is a nueral network implementation with NumPy. It can build a tape-based autograd system for neural networks learned from PyTorch. 

TripleN requires Python 3.6.0 or later.

## Components

| Component        | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| triplen          | a Tensor library like NumPy                                  |
| triplen.autograd | a tape-based automatic differentiation library that supports all differentiable Tensor operations in triple |
| triplen.nn       | a neural networks library                                    |

## Usage

You can build your networks as PyTorch Style.

```python
import triplen.nn as nn
import triplen.nn.functional as F

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()

        self.conv1 = nn.Conv2D(1, 16, 5)
        self.conv2 = nn.Conv2D(16, 32, 3)
        self.fc1 = nn.Linear(5 * 5 * 32, 200)
        self.fc2 = nn.Linear(200, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        return x

```

A series of Examples is in [examples folder](https://github.com/ThdLee/TripleN/tree/master/examples).

