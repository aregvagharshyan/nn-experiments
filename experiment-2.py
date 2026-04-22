import os
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Data generation
# =========================

np.random.seed(42)
n, d = 2000, 100
K = 1000

lambda_reg = 0.1
gamma = 1e-2

A = np.random.randn(n, d)
x_true = np.random.randn(d)
noise = 0.1 * np.random.randn(n)
y = A @ x_true + noise

# -----------------------
# Functions
# -----------------------
def loss_i(i, x):
    return (A[i] @ x - y[i])**2

def grad_i(i, x):
    return 2 * A[i] * (A[i] @ x - y[i])

def reg_grad(x):
    return lambda_reg * (2 * x / (1 + x**2)**2)

def full_loss(x):
    data_loss = np.mean([(A[i] @ x - y[i])**2 for i in range(n)])
    reg = lambda_reg * np.sum(x**2 / (1 + x**2))
    return data_loss + reg

def full_grad(x):
    g = np.zeros(d)
    for i in range(n):
        g += grad_i(i, x)
    g /= n
    g += reg_grad(x)
    return g

# =========================
# ES quantities
# =========================

def es_quantity(x):
    return np.mean([np.linalg.norm(grad_i(i, x))**2 for i in range(n)])

# =========================
# SGD run (store snapshots)
# =========================

x = np.zeros(d)

snap_losses = []
snap_es = []
snap_fg = []
snap_x = []

for k in range(K):
    i = np.random.randint(n)
    g = grad_i(i, x)

    x -= gamma * (g + reg_grad(x))

    # snapshot every 10 steps
    if k % 10 == 0:
        snap_x.append(x.copy())
        snap_losses.append(full_loss(x))
        snap_es.append(es_quantity(x))
        snap_fg.append(np.linalg.norm(full_grad(x))**2)

snap_losses = np.array(snap_losses)
snap_es = np.array(snap_es)
snap_fg = np.array(snap_fg)

# =========================
# ES vs loss-gap check
# =========================

f_star = np.min(snap_losses)
loss_gap = snap_losses - f_star

corr1 = np.corrcoef(snap_es, loss_gap)[0, 1]
corr2 = np.corrcoef(snap_es, snap_fg)[0, 1]

print("Corr(ES, loss-gap):", corr1)
print("Corr(ES, full-grad-norm):", corr2)

# =========================
# Plots
# =========================

plt.figure()
plt.plot(loss_gap, label="f(x) - f*")
plt.plot(snap_es, label="ES quantity")
plt.legend()
plt.title("ES vs Loss gap")
plt.show()

plt.figure()
plt.scatter(loss_gap, snap_es)
plt.xlabel("f(x) - f*")
plt.ylabel("E ||∇f_i(x)||^2")
plt.title("Expected Smoothness relationship")
plt.show()