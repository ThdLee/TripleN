import matplotlib.pyplot as plt
import pickle

with open('./examples/mnist/pytorch.bin', 'rb') as f:
    pytorch_data = pickle.load(f)

with open('./examples/mnist/triplen.bin', 'rb') as f:
    triplen_data = pickle.load(f)

plt.subplot(2, 1, 1)
plt.plot(pytorch_data['train_acc'][:1000], label='pytorch')
plt.plot(triplen_data['train_acc'][:1000], label='triplen')
plt.ylabel('train accuracy')
plt.xlabel('steps')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(pytorch_data['train_loss'][:1000], label='pytorch')
plt.plot(triplen_data['train_loss'][:1000], label='triplen')
plt.ylabel('train loss')
plt.xlabel('steps')
plt.legend()

# plt.subplot(2, 2, 3)
# plt.plot(pytorch_data['val_acc'])
# plt.ylabel('val_acc')
#
# plt.subplot(2, 2, 4)
# plt.plot(pytorch_data['val_test'])
# plt.ylabel('val_test')

plt.show()