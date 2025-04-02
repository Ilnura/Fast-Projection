import numpy as np
from numpy import linalg
from scipy.optimize import linprog, minimize_scalar
from scipy.stats import norm, chi2
from scipy.optimize import minimize

def f_proj(x, x_p, params):
    return np.linalg.norm(x - x_p)**2

def f_proj_grad(x, x_p, params):
    return 2. * (x - x_p)

def prox_box(low,up,x):
    return np.maximum(np.minimum(x,up),low)

def Id(low,up,x):
    return x