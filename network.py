import os
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from tensorflow.keras.datasets import mnist

# ======================
# Setup
# ======================

np.random.seed(42)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = (x_train / 255.0).astype(np.float32)[..., None]
x_test = (x_test / 255.0).astype(np.float32)[..., None]

def one_hot(y):
    return np.eye(10)[y]

y_train = one_hot(y_train)
y_test = one_hot(y_test)

# ======================
# Init (xavier safe)
# ======================

def xavier(n_in, n_out):
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_in, n_out)).astype(np.float32)

conv = np.random.randn(3,3,1,8).astype(np.float32) * 0.1

W1 = xavier(13*13*8, 64)
b1 = np.zeros((64,), dtype=np.float32)

W2 = xavier(64, 10)
b2 = np.zeros((10,), dtype=np.float32)

# ======================
# Activations
# ======================

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

# ======================
# Layers
# ======================

def conv2d(x, f):
    h, w, _ = x.shape
    fh, fw, _, n_f = f.shape

    out = np.zeros((h-2, w-2, n_f), dtype=np.float32)

    for k in range(n_f):
        for i in range(h-2):
            for j in range(w-2):
                out[i,j,k] = np.sum(x[i:i+3, j:j+3] * f[:,:,:,k])

    return out


def maxpool(x):
    h,w,c = x.shape
    out = np.zeros((h//2, w//2, c), dtype=np.float32)

    for i in range(0,h,2):
        for j in range(0,w,2):
            out[i//2,j//2] = np.max(x[i:i+2, j:j+2], axis=(0,1))

    return out

# ======================
# Cache
# ======================

cache = None

# ======================
# Forward (safe shapes)
# ======================

def forward(x):
    global cache

    zc = conv2d(x, conv)
    ac = relu(zc)
    pc = maxpool(ac)

    flat = pc.reshape(-1)

    z1 = flat @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    out = softmax(z2).reshape(-1)

    cache = (x, zc, ac, pc, flat, z1, a1, z2)

    return out

# ======================
# Loss
# ======================

def loss(pred, y):
    pred = np.clip(pred, 1e-7, 1-1e-7)
    return -np.sum(y * np.log(pred))

# ======================
# Backward (fixed)
# ======================

def backward(pred, y, lr=0.001):
    global W1,b1,W2,b2,conv

    pred = pred.reshape(-1)
    y = y.reshape(-1)

    x, zc, ac, pc, flat, z1, a1, z2 = cache

    # output gradient
    dz2 = pred - y  # (10,)

    dW2 = np.outer(a1, dz2)
    db2 = dz2.copy()

    da1 = W2 @ dz2
    dz1 = da1 * (z1 > 0)

    dW1 = np.outer(flat, dz1)
    db1 = dz1.copy()

    # update dense
    W2 -= lr * dW2
    b2 -= lr * db2

    W1 -= lr * dW1
    b1 -= lr * db1

    # conv (simplified but stable)
    dflat = W1 @ dz1
    dpc = dflat.reshape(pc.shape)

    for k in range(conv.shape[-1]):
        conv[:,:,:,k] -= lr * np.mean(dpc[:,:,k])

# ======================
# Batch training
# ======================

def get_batches(x, y, batch=32):
    for i in range(0, len(x), batch):
        yield x[i:i+batch], y[i:i+batch]

# ======================
# Train
# ======================

lr = 0.001

for epoch in range(2):
    total = 0

    for x_batch, y_batch in get_batches(x_train[:2000], y_train[:2000], 32):

        batch_loss = 0

        for x, y in zip(x_batch, y_batch):
            pred = forward(x)
            batch_loss += loss(pred, y)
            backward(pred, y, lr)

        total += batch_loss

    print("Epoch:", epoch, "Loss:", total)

# ======================
# Test
# ======================

correct = 0

for i in range(200):
    pred = forward(x_test[i])
    if np.argmax(pred) == np.argmax(y_test[i]):
        correct += 1

print("Accuracy:", correct / 200)