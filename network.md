# CNN Architecture for MNIST

Supervised training of a small convolutional neural network for MNIST digit classification, using manually implemented NumPy forward propagation and full backward propagation.

## Model Architecture

```text
Input image
28 × 28 × 1
      │
      ▼
Valid 2-D convolution
3 × 3 filters, 8 output channels
26 × 26 × 8
      │
      ▼
ReLU activation
26 × 26 × 8
      │
      ▼
2 × 2 non-overlapping max-pooling
13 × 13 × 8
      │
      ▼
Flatten
1,352 values
      │
      ▼
Fully connected layer
64 units
      │
      ▼
ReLU activation
64 values
      │
      ▼
Fully connected output layer
10 units
      │
      ▼
Softmax
10 class probabilities
```
- Total trainable parameters: **87,314**.
- The convolutional layer has no bias. The dense layers include biases.

## Forward Propagation
For an input image `x` the chain is:

```text
→ convolution
→ ReLU
→ max-pooling
→ flatten
→ dense layer
→ ReLU
→ dense output layer
→ softmax probabilities
```

The convolution uses valid cross-correlation, so a `28 × 28` image becomes `26 × 26` after applying a `3 × 3` filter.

## Loss Function
The model uses categorical cross-entropy for the correct digit label `y`:

```python
L = -log(p_y)
```

For softmax combined with cross-entropy, the output gradient is:

```python
dL/dlogits = probabilities - one_hot(label)
```

## Backward Propagation
The gradient is propagated through the layers in reverse order:

```text
cross-entropy and softmax
 → output dense layer
 → ReLU
 → hidden dense layer
 → flatten
 → max-pooling
 → ReLU
 → convolution
```

During max-pooling backpropagation, the gradient is sent only to the input value that produced each maximum.

During convolution backpropagation, every filter weight receives its own gradient:

```python
dfilters += patch[:, :, :, None] * upstream[None, None, None, :]
```

## Training Configuration
- Dataset: MNIST
- Input normalization: pixel values divided by `255.0`
- Training examples: first `2,000` images by default
- Test examples: first `200` images by default
- Optimizer: manually implemented mini-batch gradient descent
- Batch size: `32`
- Learning rate: `0.01`
- Epochs: `2`
- Weight initialization: He initialization for layers followed by ReLU


