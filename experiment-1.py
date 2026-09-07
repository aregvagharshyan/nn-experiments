# [Stochastic optimization of a regularized linear regression model with uniform, importance, and adaptive sampling.]
#

import os
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

# 
# Data generation
# 

np.random.seed(42)

n = 2000
d = 100

A = np.random.randn(n, d)
x_true = np.random.randn(d)
y = A @ x_true + 0.1 * np.random.randn(n)

#
# Objective
# 

def f(x, A, y, lam=0.1):
    return np.mean((A @ x - y)**2) + lam * np.sum(x**2 / (1 + x**2))

def grad_i(x, a_i, y_i):
    return 2 * (a_i @ x - y_i) * a_i

def grad_r(x, lam=0.1):
    return lam * (2 * x / (1 + x**2)**2)

# 
# Uniform SGD
#

def train_uniform(A, y, K=5000, lr=1e-2, lam=0.1):
    n, d = A.shape
    x = np.zeros(d)
    history = []

    for k in range(K):
        i = np.random.randint(n)

        g = grad_i(x, A[i], y[i]) + grad_r(x, lam)
        x = x - lr * g

        if k % 50 == 0:
            history.append(f(x, A, y, lam))

    return history

# 
# Importance sampling
#

def train_importance(A, y, K=5000, lr=1e-2, lam=0.1):
    n, d = A.shape
    x = np.zeros(d)
    history = []

    probs = np.sum(A**2, axis=1)
    probs = probs / np.sum(probs)

    for k in range(K):
        i = np.random.choice(n, p=probs)

        g = grad_i(x, A[i], y[i]) + grad_r(x, lam)
        x = x - lr * g

        if k % 50 == 0:
            history.append(f(x, A, y, lam))

    return history

# 
# Adaptive sampling
#

def train_stable_adaptive(A, y, K=5000, lr=1e-2, lam=0.1, beta=0.1, alpha=0.7):
    n, d = A.shape
    x = np.zeros(d)

    # smoothed weights (IMPORTANT FIX)
    weights = np.ones(n)
    history = []

    eps = 1e-12

    for k in range(K):

        # normalized probabilities (stable power law)
        w_pow = np.power(weights, alpha)
        probs = w_pow / (np.sum(w_pow) + eps)

        # safety fallback
        if np.any(np.isnan(probs)) or np.any(probs < 0):
            probs = np.ones(n) / n

        i = np.random.choice(n, p=probs)

        g = grad_i(x, A[i], y[i]) + grad_r(x, lam)
        x = x - lr * g

        # SMOOTH update (critical fix)
        g_norm = np.linalg.norm(g)**2
        weights[i] = (1 - beta) * weights[i] + beta * g_norm

        # clamp to avoid explosion
        weights[i] = np.clip(weights[i], 1e-8, 1e3)

        if k % 50 == 0:
            history.append(f(x, A, y, lam))

    return history
    
# 
# SGD run
# 

h_u = train_uniform(A, y)
h_i = train_importance(A, y)
h_a = train_stable_adaptive(A, y)

print("Final Uniform Loss:", h_u[-1])
print("Final Importance Loss:", h_i[-1])
print("Final Adaptive Loss:", h_a[-1])

# 
# Plot results
# 

plt.plot(h_u, label='Uniform SGD')
plt.plot(h_i, label='Fixed Importance Sampling')
plt.plot(h_a, label='Adaptive Sampling (Ours)')

plt.title('SGD Sampling Strategies Comparison')
plt.xlabel('iterations (x50)')
plt.ylabel('loss')
plt.legend()
plt.show()
