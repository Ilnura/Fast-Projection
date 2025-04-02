import numpy as np
from numpy import linalg
from scipy.optimize import linprog, minimize_scalar
from scipy.stats import norm, chi2
from scipy.optimize import minimize

from basic_functions import f_proj, f_proj_grad, prox_box, Id


def Ellipsoid(l_up,                                       # upper bound on the dual variable
              epsilon,                                    # target accuracy
              x_0,                                        # starting point
              x_p,                                        # projection point
              L,                                          # smoothness of the objective
              T,                                          # when to stop the dual mehtod
              algorithm,                                  # primal_method
              h_grad, h_constraint,                       # constraint functions
              learning_rate, params,                      # dont use
              f_objective = f_proj, f_grad = f_proj_grad, # objective (projection by default)
              K_max = 1000,                               # when to stop the primal method
              mu = 2.,                                    # strong convexity of the objective
              G = 1.,                                     # Lipshitz constant of the dual problem ???
              low = 0, up = 0,                            # box constraints
              prox = Id                                   # proximal function ( none by default )
              ): 
    
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
    x[0], k, iter_time = algorithm(l[0], 
                                    x_0, x_p, 
                                    t_epsilon, 
                                    epsilon, 
                                    h_grad, h_constraint, 
                                    aplha_agd, beta_agd, 
                                    learning_rate, params, 
                                    f_objective = f_objective, f_grad = f_grad, 
                                    K_max = K_max, t = 0, 
                                    low = low, up = up, prox = prox)
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
                                f_objective = f_objective, f_grad = f_grad, 
                                K_max = K_max, t = 0, 
                                low = low, up = up, prox = prox)
            
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
#         print("l_t,l_t+1 = ",l[t], l[t+1])
        if np.linalg.norm(l[t+1] - l[t]) <= epsilon ** 0.5:      
            print("converged")
            break
                    
        if abs(h[t]).dot(l[t]) <= epsilon and productive == 1:
            if np.linalg.norm(f_grad(x[t], x_p, params) \
                              + h_grad(x[t], params).T.dot(l[t])) < epsilon:
                print("crit_reached")
                break

    print(t)
    return  the_best_productive_x, the_best_productive_l, f_objective(the_best_productive_x, x_p, params)


def Bisection(l_up, epsilon, x_0, x_p, 
              L, T, algorithm, 
              h_grad, h_constraint, 
              learning_rate, params, 
              f_objective = f_proj, f_grad = f_proj_grad, low = 0, up = 0, prox = Id):
    
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
