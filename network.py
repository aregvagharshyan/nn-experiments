# [Supervised training of a CNN for MNIST classification, via manually implemented forward and backward NumPy propagation.]
#  - TensorFlow/Keras is used only to load MNIST.
#  - All model operations and gradients are implemented with NumPy.


#
# Activations
#

import os
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from tensorflow.keras.datasets import mnist


def relu(x):
    return np.maximum(x, 0.0)


def relu_backward(dout, x):
    return dout * (x > 0)


def conv2d_forward(x, filters):

    # Valid 2-D cross-correlation for one image.
    height, width, channels = x.shape
    filter_height, filter_width, filter_channels, n_filters = filters.shape
    if channels != filter_channels:
        raise ValueError("The input and filter channel counts must match")

    out_height = height - filter_height + 1
    out_width = width - filter_width + 1
    out = np.zeros((out_height, out_width, n_filters), dtype=np.float32)

    for i in range(out_height):
        for j in range(out_width):
            patch = x[i:i + filter_height, j:j + filter_width, :]

            # Each filter gets its own dot product with the input patch.
            out[i, j, :] = np.sum(
                patch[:, :, :, None] * filters,
                axis=(0, 1, 2),
            )
    return out


def conv2d_backward(x, filters, dout):

    # Backpropagate through conv2d_forward.
    # Returns gradients with respect to x and every individual filter weight.
    height, width, channels = x.shape
    filter_height, filter_width, filter_channels, n_filters = filters.shape
    out_height, out_width, _ = dout.shape

    dx = np.zeros_like(x)
    dfilters = np.zeros_like(filters)

    for i in range(out_height):
        for j in range(out_width):
            patch = x[i:i + filter_height, j:j + filter_width, :]
            upstream = dout[i, j, :]  # one value per filter

            # dL/d(filter[a,b,c,k]) = input[a,b,c] * dL/dout[i,j,k]
            dfilters += patch[:, :, :, None] * upstream[None, None, None, :]

            # dL/d(input patch) = sum_k filter[...,k] * dL/dout[i,j,k]
            dx[i:i + filter_height, j:j + filter_width, :] += np.sum(
                filters * upstream[None, None, None, :],
                axis=3,
            )

    return dx, dfilters

#
# Max-pooling
#

def maxpool2x2_forward(x):

    # Non-overlapping 2x2 max-pooling.
    # Retain each winning index for the backward pass.
    height, width, channels = x.shape
    out_height, out_width = height // 2, width // 2
    out = np.zeros((out_height, out_width, channels), dtype=np.float32)
    argmax = np.zeros((out_height, out_width, channels), dtype=np.int8)

    for i in range(out_height):
        for j in range(out_width):
            patch = x[2 * i:2 * i + 2, 2 * j:2 * j + 2, :].reshape(4, channels)
            winners = np.argmax(patch, axis=0)
            out[i, j, :] = patch[winners, np.arange(channels)]
            argmax[i, j, :] = winners

    return out, argmax


def maxpool2x2_backward(dout, argmax):

    # Send each pooling gradient to the input value that won the max.
    out_height, out_width, channels = dout.shape
    dx = np.zeros((out_height * 2, out_width * 2, channels), dtype=np.float32)

    for i in range(out_height):
        for j in range(out_width):
            for channel in range(channels):
                winner = int(argmax[i, j, channel])
                di, dj = divmod(winner, 2)
                dx[2 * i + di, 2 * j + dj, channel] = dout[i, j, channel]

    return dx

#
# CNN model
#

class SmallCNN:

    # 28x28x1 -> 3x3 convolution -> ReLU -> 2x2 max-pool -> dense layers.
    def __init__(self, seed=42, n_filters=8, hidden_units=64):
        rng = np.random.default_rng(seed)

        # "He initialization" is suitable for layers followed by ReLU.
        # "He initialization" for layers followed by ReLU.
        # "He initialization" compensates for that by using the factor 2 / n_in, helping gradients remain stable during training.
        self.conv = (
            rng.standard_normal((3, 3, 1, n_filters)).astype(np.float32)
            * np.sqrt(2.0 / 9.0)
        )
        pooled_features = 13 * 13 * n_filters
        self.W1 = (
            rng.standard_normal((pooled_features, hidden_units)).astype(np.float32)
            * np.sqrt(2.0 / pooled_features)
        )
        self.b1 = np.zeros(hidden_units, dtype=np.float32)
        self.W2 = (
            rng.standard_normal((hidden_units, 10)).astype(np.float32)
            * np.sqrt(2.0 / hidden_units)
        )
        self.b2 = np.zeros(10, dtype=np.float32)

    @staticmethod
    def softmax(logits):
        shifted = logits - np.max(logits)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials)

    def forward(self, x):

        # Forward propagation.
        # Return class probabilities and a cache for backpropagation.
        zc = conv2d_forward(x, self.conv)
        ac = relu(zc)
        pooled, pool_argmax = maxpool2x2_forward(ac)

        flat = pooled.reshape(-1)
        z1 = flat @ self.W1 + self.b1
        a1 = relu(z1)
        logits = a1 @ self.W2 + self.b2
        probabilities = self.softmax(logits)

        cache = (x, zc, pooled, pool_argmax, flat, z1, a1)
        return probabilities, cache

    def loss_and_gradients(self, probabilities, label, cache):

        # Cross-entropy loss and backward propagation
        x, zc, pooled, pool_argmax, flat, z1, a1 = cache
        loss = -np.log(np.clip(probabilities[int(label)], 1e-7, 1.0))

        # For softmax + cross-entropy, dL/dlogits = probabilities - one_hot(label).
        # For softmax + cross-entropy:
        # dL/dlogits = probabilities - one_hot(label).
        dlogits = probabilities.copy()
        dlogits[int(label)] -= 1.0

        dW2 = np.outer(a1, dlogits)
        db2 = dlogits.copy()
        da1 = self.W2 @ dlogits

        dz1 = relu_backward(da1, z1)
        dW1 = np.outer(flat, dz1)
        db1 = dz1.copy()
        dflat = self.W1 @ dz1

        dpooled = dflat.reshape(pooled.shape)
        dac = maxpool2x2_backward(dpooled, pool_argmax)
        dzc = relu_backward(dac, zc)

        # This is the important correction: one gradient per filter weight.
        # Correct convolution gradient: one gradient per filter weight.
        _, dconv = conv2d_backward(x, self.conv, dzc)

        gradients = {
            "conv": dconv,
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
        }
        return float(loss), gradients

    def update(self, gradients, learning_rate):

        # Gradient-descent update.
        self.conv -= learning_rate * gradients["conv"]
        self.W1 -= learning_rate * gradients["W1"]
        self.b1 -= learning_rate * gradients["b1"]
        self.W2 -= learning_rate * gradients["W2"]
        self.b2 -= learning_rate * gradients["b2"]

    def train(self, images, labels, epochs=2, batch_size=32, learning_rate=0.01):

        # Mini-batch gradient descent; gradients are averaged per batch.
        rng = np.random.default_rng(123)
        n_examples = len(images)

        for epoch in range(epochs):
            order = rng.permutation(n_examples)
            total_loss = 0.0
            seen = 0

            for start in range(0, n_examples, batch_size):
                batch_indices = order[start:start + batch_size]
                batch_gradients = {
                    "conv": np.zeros_like(self.conv),
                    "W1": np.zeros_like(self.W1),
                    "b1": np.zeros_like(self.b1),
                    "W2": np.zeros_like(self.W2),
                    "b2": np.zeros_like(self.b2),
                }

                for index in batch_indices:
                    probabilities, cache = self.forward(images[index])
                    sample_loss, sample_gradients = self.loss_and_gradients(
                        probabilities, labels[index], cache
                    )
                    total_loss += sample_loss
                    seen += 1
                    for name in batch_gradients:
                        batch_gradients[name] += sample_gradients[name]

                batch_count = len(batch_indices)
                for name in batch_gradients:
                    batch_gradients[name] /= batch_count
                self.update(batch_gradients, learning_rate)

            print(f"Epoch {epoch + 1}/{epochs} - loss: {total_loss / seen:.4f}")

    def predict(self, images):

        # Predict the most likely digit for each image.
        predictions = []
        for image in images:
            probabilities, _ = self.forward(image)
            predictions.append(np.argmax(probabilities))
        return np.asarray(predictions)

#
# Data loading and training
#

def main():

    # This is the only non-NumPy part: loading the standard MNIST dataset.
    np.random.seed(42)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = (x_train / 255.0).astype(np.float32)[..., None]
    x_test = (x_test / 255.0).astype(np.float32)[..., None]

    # Small subset for a quick educational run.
    # Remove the slices to train on all 60,000 MNIST training images.
    train_limit = 2_000
    test_limit = 200

    model = SmallCNN(seed=42)
    model.train(
        x_train[:train_limit],
        y_train[:train_limit],
        epochs=2,
        batch_size=32,
        learning_rate=0.01,
    )

    predictions = model.predict(x_test[:test_limit])
    accuracy = np.mean(predictions == y_test[:test_limit])
    print(f"Accuracy on {test_limit} test images: {accuracy:.2%}")


if __name__ == "__main__":
    main()
