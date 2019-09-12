import numpy as np


def linear(x, w, theta=0):
    return (x*w).sum() + theta


def sat(x, w, theta=0):
    val = (x*w).sum() + theta
    val = 1 if val > 1 else val
    val = -1 if val < -1 else val
    return val


def sig(x, w, theta=0):
    val = 1+np.exp(-(x*w).sum() + theta)
    return 1/val


def step(x, w, theta=0):
    val = (x*w).sum() + theta
    val = 1 if val > 0 else 0
    return val
