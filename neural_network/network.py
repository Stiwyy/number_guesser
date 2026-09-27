import numpy as np


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def cross_entropy_loss(predictions, labels):
    n = labels.shape[0]
    p = np.clip(predictions[np.arange(n), labels], 1e-12, 1.0)
    return -np.mean(np.log(p))


def accuracy(predictions, labels):
    return np.mean(np.argmax(predictions, axis=1) == labels)


def im2col(x, k, padding=0, stride=1):
    n, h, w, c = x.shape

    if padding:
        x = np.pad(
            x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode="constant"
        )

    oh = (h + 2 * padding - k) // stride + 1
    ow = (w + 2 * padding - k) // stride + 1

    shape = (n, oh, ow, k, k, c)
    strides = (
        x.strides[0],
        x.strides[1] * stride,
        x.strides[2] * stride,
        x.strides[1],
        x.strides[2],
        x.strides[3],
    )

    cols = np.lib.stride_tricks.as_strided(
        x, shape=shape, strides=strides, writeable=False
    )

    return cols.reshape(n * oh * ow, k * k * c), oh, ow


def col2im(cols, input_shape, k, padding=0, stride=1):
    n, h, w, c = input_shape

    oh = (h + 2 * padding - k) // stride + 1
    ow = (w + 2 * padding - k) // stride + 1

    cols = cols.reshape(n, oh, ow, k, k, c)

    out = np.zeros((n, h + 2 * padding, w + 2 * padding, c), dtype=cols.dtype)

    for i in range(k):
        for j in range(k):
            out[:, i : i + stride * oh : stride, j : j + stride * ow : stride, :] += (
                cols[:, :, :, i, j, :]
            )

    if padding:
        return out[:, padding:-padding, padding:-padding, :]

    return out


def max_pool2x2(x):
    n, h, w, c = x.shape

    h2 = h // 2
    w2 = w // 2

    x = x[:, : h2 * 2, : w2 * 2, :]

    x = x.reshape(n, h2, 2, w2, 2, c)

    out = x.max(axis=(2, 4))

    mask = x == out[:, :, None, :, None, :]

    return out, mask


def max_pool2x2_backward(dout, mask):
    dx = np.zeros(mask.shape, dtype=dout.dtype)

    dx += mask * dout[:, :, None, :, None, :]

    n, h, _, w, _, c = dx.shape

    return dx.reshape(n, h * 2, w * 2, c)


class Conv2D:
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1):
        self.kernel_size = kernel_size
        self.padding = padding

        fan_in = kernel_size * kernel_size * input_channels

        scale = np.sqrt(2.0 / fan_in)

        self.W = (
            np.random.randn(
                kernel_size, kernel_size, input_channels, output_channels
            ).astype(np.float32)
            * scale
        )

        self.b = np.zeros(output_channels, dtype=np.float32)

        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)

        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x

        self.cols, oh, ow = im2col(x, self.kernel_size, self.padding)

        w = self.W.reshape(-1, self.W.shape[-1])

        out = self.cols @ w + self.b

        return out.reshape(x.shape[0], oh, ow, self.W.shape[-1])

    def backward(self, dout):
        dout_flat = dout.reshape(-1, self.W.shape[-1])

        self.dW = (self.cols.T @ dout_flat).reshape(self.W.shape)

        self.db = np.sum(dout_flat, axis=0)

        w = self.W.reshape(-1, self.W.shape[-1])

        dcols = dout_flat @ w.T

        return col2im(dcols, self.x.shape, self.kernel_size, self.padding)

    def update(self, lr, beta1, beta2, epsilon, step):
        self.mW = beta1 * self.mW + (1.0 - beta1) * self.dW

        self.vW = beta2 * self.vW + (1.0 - beta2) * (self.dW * self.dW)

        self.mb = beta1 * self.mb + (1.0 - beta1) * self.db

        self.vb = beta2 * self.vb + (1.0 - beta2) * (self.db * self.db)

        mW_hat = self.mW / (1.0 - beta1**step)

        vW_hat = self.vW / (1.0 - beta2**step)

        mb_hat = self.mb / (1.0 - beta1**step)

        vb_hat = self.vb / (1.0 - beta2**step)

        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + epsilon)

        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + epsilon)


class Dense:
    def __init__(self, input_size, output_size):
        scale = np.sqrt(2.0 / input_size)

        self.W = np.random.randn(input_size, output_size).astype(np.float32) * scale

        self.b = np.zeros(output_size, dtype=np.float32)

        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)

        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)

        return dout @ self.W.T

    def update(self, lr, beta1, beta2, epsilon, step):
        self.mW = beta1 * self.mW + (1.0 - beta1) * self.dW

        self.vW = beta2 * self.vW + (1.0 - beta2) * (self.dW * self.dW)

        self.mb = beta1 * self.mb + (1.0 - beta1) * self.db

        self.vb = beta2 * self.vb + (1.0 - beta2) * (self.db * self.db)

        mW_hat = self.mW / (1.0 - beta1**step)

        vW_hat = self.vW / (1.0 - beta2**step)

        mb_hat = self.mb / (1.0 - beta1**step)

        vb_hat = self.vb / (1.0 - beta2**step)

        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + epsilon)

        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + epsilon)


class NeuralNetwork:
    INPUT_SIZE = 64

    def __init__(self):
        self.conv1 = Conv2D(1, 8, 3, 1)

        self.conv2 = Conv2D(8, 16, 3, 1)

        self.conv3 = Conv2D(16, 32, 3, 1)

        self.dense1 = Dense(8 * 8 * 32, 128)

        self.dense2 = Dense(128, 10)

        self.step = 0

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    def forward(self, X):
        X = X.reshape(-1, 64, 64, 1)

        x = self.conv1.forward(X)
        self.r1 = relu(x)

        x, self.m1 = max_pool2x2(self.r1)

        x = self.conv2.forward(x)
        self.r2 = relu(x)

        x, self.m2 = max_pool2x2(self.r2)

        x = self.conv3.forward(x)
        self.r3 = relu(x)

        x, self.m3 = max_pool2x2(self.r3)

        self.flat_shape = x.shape

        x = x.reshape(x.shape[0], -1)

        x = self.dense1.forward(x)
        self.r4 = relu(x)

        x = self.dense2.forward(x)

        self.output = softmax(x)

        return self.output

    def backward(self, labels):
        n = labels.shape[0]

        dz = self.output.copy()

        dz[np.arange(n), labels] -= 1

        dz /= n

        dz = self.dense2.backward(dz)

        dz *= self.r4 > 0

        dz = self.dense1.backward(dz)

        dz = dz.reshape(self.flat_shape)

        dz = max_pool2x2_backward(dz, self.m3)

        dz *= self.r3 > 0

        dz = self.conv3.backward(dz)

        dz = max_pool2x2_backward(dz, self.m2)

        dz *= self.r2 > 0

        dz = self.conv2.backward(dz)

        dz = max_pool2x2_backward(dz, self.m1)

        dz *= self.r1 > 0

        self.conv1.backward(dz)

    def update(self, learning_rate=0.001):
        self.step += 1

        self.conv1.update(
            learning_rate, self.beta1, self.beta2, self.epsilon, self.step
        )

        self.conv2.update(
            learning_rate, self.beta1, self.beta2, self.epsilon, self.step
        )

        self.conv3.update(
            learning_rate, self.beta1, self.beta2, self.epsilon, self.step
        )

        self.dense1.update(
            learning_rate, self.beta1, self.beta2, self.epsilon, self.step
        )

        self.dense2.update(
            learning_rate, self.beta1, self.beta2, self.epsilon, self.step
        )

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def save(self, path):
        np.savez(
            path,
            conv1_W=self.conv1.W,
            conv1_b=self.conv1.b,
            conv2_W=self.conv2.W,
            conv2_b=self.conv2.b,
            conv3_W=self.conv3.W,
            conv3_b=self.conv3.b,
            dense1_W=self.dense1.W,
            dense1_b=self.dense1.b,
            dense2_W=self.dense2.W,
            dense2_b=self.dense2.b,
        )

    def load(self, path):
        data = np.load(path)

        self.conv1.W = data["conv1_W"]
        self.conv1.b = data["conv1_b"]

        self.conv2.W = data["conv2_W"]
        self.conv2.b = data["conv2_b"]

        self.conv3.W = data["conv3_W"]
        self.conv3.b = data["conv3_b"]

        self.dense1.W = data["dense1_W"]
        self.dense1.b = data["dense1_b"]

        self.dense2.W = data["dense2_W"]
        self.dense2.b = data["dense2_b"]
