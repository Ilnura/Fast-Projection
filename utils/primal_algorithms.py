import numpy as np
from numpy import linalg
from scipy.optimize import linprog, minimize_scalar
from scipy.stats import norm, chi2
from scipy.optimize import minimize
from time import time
from basic_functions import f_proj, f_proj_grad, prox_box, Id


def AGD_m(l_t, x_t0, x_p, t_epsilon, epsilon, 
          h_grad, h_constraint, 
          alpha_agd, beta_agd, 
          learning_rate, params,
          f_objective = f_proj, f_grad = f_proj_grad,  
          K_max = 1000, t=1, 
          low=0, up=0, prox=Id):
    
    x_k = x_t0
    z_k = x_t0
    k = 0

    l_norm = np.linalg.norm(l_t)
    size = x_k.size
    
    kappa = beta_agd / alpha_agd
    
    u = (kappa**0.5 - 1.) / (kappa**0.5 + 1.) 
    h_gr = h_grad(x_k, params)
    
    time1 = time()
    grad = f_grad(x_k, x_p, params) + h_gr.T.dot(l_t)#l_t.dot(h_gr)
    iter_time1 = time() - time1
    grad_norm = np.linalg.norm(grad)
    diff = 100.
#     while  grad_norm > t_epsilon:
    while  diff > t_epsilon:
#         z_kPlus1 = x_k - learning_rate * grad / l_norm  
        z_kPlus1 = prox(low, up, x_k - 1./ beta_agd * grad)
        x_kPlus1 = (1 + u) * z_kPlus1 - u * z_k
        
        diff = np.linalg.norm(x_kPlus1 - x_k)
        
        x_k = x_kPlus1
        z_k = z_kPlus1
        
        h_gr = h_grad(x_k, params)
        grad = f_grad(x_k, x_p, params) + h_gr.T.dot(l_t)#l_t.dot(h_gr)
        grad_norm = np.linalg.norm(grad)

        k += 1
        
        if k > K_max:
            print("long")
            return x_k, k, 1.   
#     print(k)
    return x_k, k, 1.

def LIBRARY_m(l_t, x_t0, x_p, 
              t_epsilon, epsilon, 
              h_grad, h_constraint, 
              learning_rate, params, 
              f_objective, f_grad, t, low=0, up=0, prox=Id):
    
    fun = lambda x: f_objective(x, x_p, params) + l_t.dot(h_constraint(x, params))
    result = minimize(fun, 
                      x_t0, 
                      method='BFGS',
                      jac = lambda x: f_grad(x, x_p, params) + h_grad(x, params).T.dot(l_t),
                      tol = t_epsilon)

    res = result.x
    return res, 0

