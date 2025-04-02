import numpy as np
from numpy import linalg
from scipy.optimize import linprog, minimize_scalar
from scipy.stats import norm, chi2
from scipy.optimize import minimize

def f_proj(x, x_p, params):
    return np.linalg.norm(x - x_p)**2

def f_proj_grad(x, x_p, params):
    return 2. * (x - x_p)

### Algorithm 1 (Dual, Bisection) 

def Bisection(l_up, epsilon, x_0, x_p, L, T, algorithm, h_grad, h_constraint, 
              learning_rate, params, 
              f_objective = f_proj, f_grad = f_proj_grad):
    
    d = x_p.size
    l_min = 0.
    l_max = l_up
    
    t_epsilon = epsilon**2 / 2. / L**2 
    
    l = np.zeros(T)
    h = np.zeros(T)
    x = np.zeros((T,d))
    
    
    for t in range(T):
        if t == 0:
            x_prev = x_0
        else:
            x_prev = x[t - 1]
            
        l[t] = (l_max + l_min) / 2.
        
        x[t], k = algorithm(l[t], x_prev, x_p, t_epsilon, epsilon, h_grad, h_constraint, 
                       learning_rate, params, f_objective, f_grad, t)
        h[t] = h_constraint(x[t], params)

        if abs(h[t]) <= epsilon:
            return x[t], l[t], h[t]
            break

        elif (h[t]) > epsilon:
            l_min = l[t]
        else:
            l_max = l[t]
        
    if abs(h[t] * l[t]) < epsilon and np.linalg.norm(f_grad(x[t], x_p, params) 
                                                         + h_grad(x[t], params) * l[t])< epsilon:         
        print("crit reached ")
    else:
        print("crit not reached")
        
    ft = f_objective(x[t], x_p, params)    
    
    return x[t], l[t], h[t], ft


### Algorithm 2 (Primal, Gradient Oracle, GD ) 

def GD(l_t, x_t0, x_p, t_epsilon, epsilon, h_grad, h_constraint, learning_rate, params, f_objective =  f_proj, f_grad = f_proj_grad, t = 1):
    
    x_k = x_t0
    k = 0
    
    
    while  np.linalg.norm(f_grad(x_k, x_p, params) + l_t * h_grad(x_k, params)) > t_epsilon:
        grad = f_grad(x_k, x_p, params) + l_t * h_grad(x_k, params)
        x_k = x_k - learning_rate / l_t * grad
        k += 1
        if k > 10000: 
            print("inf")
            break
    return x_k, k

### Algorithm 2 (Primal, Gradient Oracle, AGD ) 

def AGD(l_t, x_t0, x_p, t_epsilon, epsilon, 
        h_grad, h_constraint, learning_rate, params, 
        f_objective=f_proj, f_grad=f_proj_grad, t=1):

    x_k = x_t0
    z_k = x_t0
    k = 0
    
    kappa = (1./learning_rate * l_t)/ (min(1./ learning_rate * l_t, 0.002))
    u = (kappa**0.5 - 1.)/(kappa**0.5 + 1.) 
    
    grad = f_grad(x_k, x_p, params) + h_grad(x_k, params) * l_t
    grad_norm = np.linalg.norm(grad)
    
    size = x_k.size
    while  grad_norm > t_epsilon:
        z_kPlus1 = x_k - learning_rate * grad /  l_t  
        x_kPlus1 = (1 + u) * z_kPlus1 - u * z_k       
        x_k = x_kPlus1
        z_k = z_kPlus1
        
        grad = f_grad(x_k, x_p, params) + h_grad(x_k, params) * l_t
        grad_norm = np.linalg.norm(grad)     
        k += 1
        
        if k > 10000:
            print("inf loop", t)
            break
    return x_k, k


## Algorithm 2 library
def LIBRARY(l_t, x_t0, x_p, t_epsilon, epsilon, h_grad, h_constraint, learning_rate, params, f_objective=f_proj, f_grad=f_proj_grad, t=1):
    fun = lambda x: f_objective(x,x_p,params) + l_t * h_constraint(x,params)
    res = minimize(fun, 
                   x_t0, 
                   method='BFGS', 
                   jac=lambda x:f_grad(x,x_p,params) +  l_t * h_grad(x,params),
                   tol=t_epsilon).x
    return res, 0

##########################################################################################################################


### Algorithm 1 Multiconstraints (Dual, Ellipsoid) 

def Ellipsoid(l_up, epsilon, x_0, x_p, L, T, algorithm, h_grad, h_constraint, learning_rate, params, 
              f_objective = f_proj, f_grad = f_proj_grad, K_max = 1000, mu = 0.02, G = 1.):
    
    d = x_p.size
    m = h_constraint(x_p, params).size
    l_min = 0
    l_max = l_up
    
    l = np.zeros((T,m))
    h = np.zeros((T,m))
    x = np.zeros((T,d))
    g = np.ones(T)*(-500.)
    degenerate = np.zeros(m)
    
    #parameters of ellipsoid method
    alpha = (m**2 / (m**2 - 1))**0.5
    gamma = (1 - ((m - 1) / (m + 1))**0.5) * alpha
    ka =  m**2 / (m**2 - 1) * ((m - 1) / (m + 1))**0.5
    beta = 1.
    t_max = 0
    
    B = np.eye(m) * l_max * m**0.5 / 2.
    l[0] = l_max / 2. * np.ones(m)
    the_best_productive_l = l[0]
    l_norm = np.linalg.norm(l[0])
    t_epsilon = epsilon**2 / 2. / G**2 
    aplha_agd = mu
    beta_agd = mu + l_norm * L
    x[0], k, iter_time = algorithm(l[0], x_0, x_p, 
                                t_epsilon, epsilon, 
                                h_grad, h_constraint, 
                                aplha_agd, beta_agd, learning_rate, params, 
                                f_objective, f_grad, K_max, 0)
    the_best_productive_x = x[0] 
    h[0] = h_constraint(x[0], params)
    e_t = h[0]
    p = B.T.dot(e_t) / np.linalg.norm(B.T.dot(e_t))
    l[1] = l[0] + 1. / (m + 1) * B.dot(p)
    B = alpha * B - gamma * np.outer(B.dot(p), p)

    for i in range(m):       
        if abs(l[1, i]) < epsilon:
            l[1, i] = 0.
            degenerate[i] = 1
    
    for t in range(1, T-1):
        t_productive = 0.
        
        if (l[t] >= 0).all() and (l[t] < l_max).all():
            productive = 1
            t_productive += 1
#             if t == 0:
#                 x0 = x_0
#             else:
            x0 = x[t - 1]
            
            l_norm = np.linalg.norm(l[t])
            t_epsilon = epsilon**2 / 2. / G**2 
            aplha_agd = mu
            beta_agd = mu + l_norm * L

#           kappa = (1./learning_rate * l_norm)/ (min(1./ learning_rate * l_norm, 0.002))
            
            x[t], k, iter_time = algorithm(l[t], x0, x_p, 
                                t_epsilon, epsilon, 
                                h_grad, h_constraint, 
                                aplha_agd, beta_agd, learning_rate, params, 
                                f_objective, f_grad, K_max, t)
            
            h[t] = h_constraint(x[t], params)
            e_t = h[t]
            if f_objective(x[t],x_p,params) + h[t].dot(l[t]) >= \
            f_objective(the_best_productive_x,x_p,params) + \
            h_constraint(the_best_productive_x, params).dot(the_best_productive_l):
                the_best_productive_x  = x[t]
                the_best_productive_l = l[t]
#                 print("better")

            for i in range(m):
                if degenerate[i] == 1:
                    e_t[i] = 0.
        
        else:
            productive = -1
#             print("not productive")
            e = np.zeros(m)
            for i in range(m):
                if l[t,i] < 0:
                    e[i] = 1
                elif l[t,i] > l_max:
                    e[i] = -1
                else:
                    e[i] = 0
#             t = t - 1 
            e_t = e
            
        p = B.T.dot(e_t) / np.linalg.norm(B.T.dot(e_t))
        l[t + 1] = l[t] + 1. / (m + 1) * B.dot(p)
                     
        B = alpha * B - gamma * np.outer(B.dot(p), p)

        for i in range(m):       
            if abs(l[t + 1, i]) < epsilon:
                l[t + 1, i] = 0.
                degenerate[i] = 1
        if abs(h[t]).dot(l[t]) <= epsilon * 2. and productive == 1:
            if np.linalg.norm(f_grad(x[t], x_p, params) + h_grad(x[t], params).T.dot(l[t])) < epsilon:
                print("crit_reached")
                #         if ka**t < epsilon * 0.5:
#                 print(t)
#                 return  x[t], l[t], f_objective(x[t], x_p, params)
                break
#     return  x[t], l[t], f_objective(x[t], x_p, params)#, x[t_max], l[t_max], f_objective(x[t_max], x_p, params)
    print(t)
    return  the_best_productive_x, the_best_productive_l, f_objective(the_best_productive_x, x_p, params)


### Algorithm 2 Multiconstraint(Primal, Gradient Oracle, GD ) 

def GD_m(l_t, x_t0, x_p, t_epsilon, epsilon, h_grad, h_constraint, learning_rate, params, f_objective=f_proj, f_grad=f_proj_grad, t=1):
    
    x_k = x_t0
    k = 0
    grad = f_grad(x_k, x_p, params) + h_grad(x_k, params).transpose().dot(l_t)
    grad_norm = np.linalg.norm(grad)
    
    while  grad_norm > t_epsilon:
          
        x_k = x_k - learning_rate / np.linalg.norm(l_t) * grad
        
        grad = f_grad(x_k, x_p, params) + h_grad(x_k, params).transpose().dot(l_t)
        grad_norm = np.linalg.norm(grad)
        k += 1
        
        if k > 10000:
            break
    return x_k, k


### Algorithm 2 Multiconstraint(Primal, Gradient Oracle, AGD ) 
import time

def AGD_m_3(l_t, x_t0, x_p, t_epsilon, epsilon, 
          h_grad, h_constraint, 
          alpha_agd, beta_agd, learning_rate, params,
          f_objective=f_proj, f_grad=f_proj_grad,  K_max = 1000, t=1):
    
    x_k = x_t0
    z_k = x_t0
    k = 0

    l_norm = np.linalg.norm(l_t)
    size = x_k.size
    
#     kappa = beta_agd / alpha_agd
    kappa = (1./learning_rate * l_norm)/ (min(1./ learning_rate * l_norm, 0.002))
    
    u = (kappa**0.5 - 1.) / (kappa**0.5 + 1.) 
    h_gr = h_grad(x_k, params)
    
    time1 = time.time()
    grad = f_grad(x_k, x_p, params) + h_gr[0] * l_t[0] + h_gr[1] * l_t[1]
    iter_time1 = time.time() - time1
    grad_norm = np.linalg.norm(grad)

    while  grad_norm > t_epsilon:
#         z_kPlus1 = x_k - learning_rate * grad / l_norm  
        z_kPlus1 = x_k - 1./ beta_agd * grad
        x_kPlus1 = (1 + u) * z_kPlus1 - u * z_k

        x_k = x_kPlus1
        z_k = z_kPlus1
        
        h_gr = h_grad(x_k, params)
        grad = f_grad(x_k, x_p, params) + h_gr[0] * l_t[0] + h_gr[1] * l_t[1]
        grad_norm = np.linalg.norm(grad)

        k += 1
        
        if k > K_max:
            print("long")
            return x_k, k, 1.   
    print(k)
    return x_k, k, 1.

def AGD_m_2(l_t, x_t0, x_p, t_epsilon, epsilon, 
          h_grad, h_constraint, 
          alpha_agd, beta_agd, learning_rate, params,
          f_objective=f_proj, f_grad=f_proj_grad,  K_max = 1000, t=1):
    
    x_k = x_t0
    z_k = x_t0
    k = 0

    l_norm = np.linalg.norm(l_t)
    size = x_k.size
    
    kappa = beta_agd / alpha_agd
#     kappa = (1./learning_rate * l_norm)/ (min(1./ learning_rate * l_norm, 0.002))
    
    u = (kappa**0.5 - 1.) / (kappa**0.5 + 1.) 
    h_gr = h_grad(x_k, params)
    
    time1 = time.time()
    grad = f_grad(x_k, x_p, params) + h_gr[0] * l_t[0] + h_gr[1] * l_t[1]
    iter_time1 = time.time() - time1
    grad_norm = np.linalg.norm(grad)

    while  grad_norm > t_epsilon:
#         z_kPlus1 = x_k - learning_rate * grad / l_norm  
        z_kPlus1 = x_k - 1./ beta_agd * grad
        x_kPlus1 = (1 + u) * z_kPlus1 - u * z_k

        x_k = x_kPlus1
        z_k = z_kPlus1
        
        h_gr = h_grad(x_k, params)
        grad = f_grad(x_k, x_p, params) + h_gr[0] * l_t[0] + h_gr[1] * l_t[1]
        grad_norm = np.linalg.norm(grad)

        k += 1
        
        if k > K_max:
            print("long")
            return x_k, k, 1.   
    print(k)
    return x_k, k, 1.

def AGD_m(l_t, x_t0, x_p, t_epsilon, epsilon, 
          h_grad, h_constraint, 
          alpha_agd, beta_agd, learning_rate, params,
          f_objective=f_proj, f_grad=f_proj_grad,  K_max = 1000, t=1):
    
    x_k = x_t0
    z_k = x_t0
    k = 0

    l_norm = np.linalg.norm(l_t)
    size = x_k.size
    
    kappa = beta_agd / alpha_agd
#     kappa = (1./learning_rate * l_norm)/ (min(1./ learning_rate * l_norm, 0.002))
    
    u = (kappa**0.5 - 1.) / (kappa**0.5 + 1.) 
    h_gr = h_grad(x_k, params)
    
    time1 = time.time()
    grad = f_grad(x_k, x_p, params) + h_gr.T.dot(l_t)#l_t.dot(h_gr)
    iter_time1 = time.time() - time1
    grad_norm = np.linalg.norm(grad)

    while  grad_norm > t_epsilon:
#         z_kPlus1 = x_k - learning_rate * grad / l_norm  
        z_kPlus1 = x_k - 1./ beta_agd * grad
        x_kPlus1 = (1 + u) * z_kPlus1 - u * z_k

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

## Algorithm 2 library

def LIBRARY_m(l_t, x_t0, x_p, 
              t_epsilon, epsilon, 
              h_grad, h_constraint, 
              learning_rate, params, 
              f_objective, f_grad, t):
    
    fun = lambda x: f_objective(x, x_p, params) + l_t.dot(h_constraint(x, params))
    result = minimize(fun, 
                      x_t0, 
                      method='BFGS',
                      jac = lambda x: f_grad(x, x_p, params) + h_grad(x, params).T.dot(l_t),
                      tol = t_epsilon)

    res = result.x
    return res, 0



