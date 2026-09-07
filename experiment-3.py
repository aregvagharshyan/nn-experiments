# [Stochastic optimization of a linear model]
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

n = 2000
d = 100

A = np.random.randn(n, d)
x_true = np.random.randn(d)
noise = 0.1 * np.random.randn(n)

y = np.sign(A @ x_true + noise)
y[y == 0] = 1


#
# Logistic loss + regularizer
# 

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def grad_i(x, i):
    a_i = A[i]
    y_i = y[i]

    z = y_i * (a_i @ x)
    coeff = -y_i * (1 - sigmoid(z))

    grad_logistic = coeff * a_i

    # non-convex regularizer gradient
    reg_grad = 2 * x / (1 + x ** 2) ** 2

    return grad_logistic + (0.1 * reg_grad)


def full_grad(x):
    g = np.zeros_like(x)
    for i in range(n):
        g += grad_i(x, i)
    return g / n


# 
# ES computation
# 

def ES(x):
    val = 0.0
    for i in range(n):
        g = grad_i(x, i)
        val += np.linalg.norm(g) ** 2
    return val / n


# 
# Entropy
# 

def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p))


# 
# Sampling strategies
# 

# Uniform
def sample_uniform():
    return np.random.randint(n)


# Importance sampling (fixed)
a_norms = np.sum(A ** 2, axis=1)
p_importance = a_norms / np.sum(a_norms)


def sample_importance():
    return np.random.choice(n, p=p_importance)


# Adaptive sampling
w = np.ones(n)

beta = 0.9
alpha = 0.7


def update_weights(x):
    global w
    for i in range(n):
        g = grad_i(x, i)
        w[i] = (1 - beta) * w[i] + beta * np.linalg.norm(g) ** 2


def sample_adaptive():
    p = w ** alpha
    p = p / np.sum(p)
    return np.random.choice(n, p=p)


# 
# SGD experiment runner
#

def run(method="uniform", T=2000, gamma=1e-2):
    x = np.zeros(d)

    ES_log = []
    loss_log = []
    grad_norm_log = []
    entropy_log = []

    global w
    w = np.ones(n)

    for t in range(T):

        # sampling
        if method == "uniform":
            i = sample_uniform()
        elif method == "importance":
            i = sample_importance()
        elif method == "adaptive":
            update_weights(x)
            i = sample_adaptive()

        # SGD step
        g = grad_i(x, i)
        x = x - gamma * g

        # logging every 50 iterations
        if t % 50 == 0:

            full_g = full_grad(x)

            ES_log.append(ES(x))
            loss = np.mean(np.log(1 + np.exp(-y * (A @ x))))
            loss_log.append(loss)
            grad_norm_log.append(np.linalg.norm(full_g) ** 2)

            if method == "adaptive":
                p = w ** alpha
                p = p / np.sum(p)
                entropy_log.append(entropy(p))
            else:
                entropy_log.append(entropy(np.ones(n) / n))

    return ES_log, loss_log, grad_norm_log, entropy_log


# 
# Run experiments
# 

ES_u, loss_u, gn_u, ent_u = run("uniform")
ES_i, loss_i, gn_i, ent_i = run("importance")
ES_a, loss_a, gn_a, ent_a = run("adaptive")

# 
# Helper for x-axis
# 

def x_axis(log):
    return np.arange(len(log)) * 50


# 
# ES comparison plot
# 

plt.figure()
plt.plot(x_axis(ES_u), ES_u, label='Uniform')
plt.plot(x_axis(ES_i), ES_i, label='Importance')
plt.plot(x_axis(ES_a), ES_a, label='Adaptive')

plt.title('Expected Smoothness (ES) over iterations')
plt.xlabel('Iteration')
plt.ylabel('ES')
plt.legend()
plt.grid(True)
plt.show()


#
# Loss plot
# 

plt.figure()
plt.plot(x_axis(loss_u), loss_u, label='Uniform')
plt.plot(x_axis(loss_i), loss_i, label='Importance')
plt.plot(x_axis(loss_a), loss_a, label='Adaptive')

plt.title('Loss over iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# 
# Gradient norm plot
# 

plt.figure()
plt.plot(x_axis(gn_u), gn_u, label='Uniform')
plt.plot(x_axis(gn_i), gn_i, label='Importance')
plt.plot(x_axis(gn_a), gn_a, label='Adaptive')

plt.title('Full Gradient Norm over iterations')
plt.xlabel('Iteration')
plt.ylabel('||∇f(x)||^2')
plt.legend()
plt.grid(True)
plt.show()


# 
# Entropy plot (key for adaptive collapse)
# 

plt.figure()
plt.plot(x_axis(ent_u), ent_u, label='Uniform')
plt.plot(x_axis(ent_i), ent_i, label='Importance')
plt.plot(x_axis(ent_a), ent_a, label='Adaptive')

plt.title('Sampling Entropy over iterations')
plt.xlabel('Iteration')
plt.ylabel('Entropy')
plt.legend()
plt.grid(True)
plt.show()
