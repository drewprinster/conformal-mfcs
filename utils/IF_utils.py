import imp
import logging
imp.reload(logging)
logging.basicConfig(level=logging.INFO)

from functools import partial

import autograd
import autograd.numpy as np
import matplotlib.pyplot as plt
import stan
import seaborn as sns

from scipy.optimize import minimize

from utils import bayesnn

## Added by Drew
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np ## Later ideally convert everything to jnp ?
import scipy
import math
import tqdm
from datetime import datetime
from random import sample

'''
Function : Gradient function G(*, omega) as in Algorithm 2 (Giordano et al. 2019)

Parameters: 
- likel(params, X, y, weights) := (Log) likelihood function for all the data (returns scalar value)
- params                       := Theta parameters
- X                            := Training data covariates
- y                            := Training data labels
- weights                      := Weight values at which to evaluate gradient
- omega                        := Omega indicator of whether to take directional derivative with respect to weights
- delta_w                      := Direction of weights (change in weights) along which to evaluate directional derivative, if applicable

Output:
- GG(*, omega) \in \mathbb{R}^D := Gradient of the likelihood loss with respect to theta parameters, evaluated at weights
'''
def get_gradient_func(loss, X, y, weights, omega):
    ## Compute derivative with respect to parameters
    if (omega == 0):
        def gradient_func(theta_0):
            return autograd.grad(loss, 0)(theta_0, X, y, np.ones(len(weights))) ## Note: The 0 argument not necessary, reminds that gradient wrt params
        
    
    ## Else omega == 1: Compute directional derivative with respect to weights in direction of (weights - np.ones(len(weights)))
    ## See bottom of pg 4 in Giordano et al. 2019 for why can implement as follows
    else:
        def gradient_func(theta_0):
            return autograd.grad(loss, 0)(theta_0, X, y, weights) - autograd.grad(loss, 0)(theta_0, X, y, np.ones(len(weights)))

    return gradient_func 
    

'''
Algorithm 1 in Giordano et al. 2019 : Forward-mode automatic differentiation

Parameters: 
- f() : \mathbb{R}^D -> \mathbb{R}^D := (Gradient) function to differentiate
- v \in \mathbb{R}^D                 := Direction along which to evlauate directional derivative / project derivative

Output:
- \partial{f(\theta)} / \partial{\theta} |_{\theta_o} v := Directional derivative of f with respect to \theta
   evaluated at \theta_0 in the direction of v (projected onto v)
'''

def ForwardModeAD(f, v):
    def FMAD_func(theta_0):
        return autograd.make_jvp(f)(theta_0)(v)[1]
    
    return FMAD_func

    
'''
Algorithm 2 in Giordano et al. 2019 : Evaluate \tau(\mathbb{K}, \omega, w)

Parameters (See Definition 2):
- \mathbb{K} := Size |\mathbb{K}| (possibly empty) set of positive integers
- \omega     := Indicator of whether to take directional derivative with respect to w
- \Theta_{max_{k \in \mathbb{K}}(k)} := Set of directional derivatives of \hat{\theta(w)} order less than or equal to k
                                        Organized so that index 
- w          := Location (in the space of data weights) where the derivative is evaluated

Output:
- 
'''
def EvaluateTerm(K, omega, Theta_k, theta_hat, loss, X, y, weights):
#     print("theta_hat len : ", len(theta_hat))
#     if (len(Theta_k) > 0):
#         print("Theta_k[0] len = ", len(Theta_k[0]))
    f = get_gradient_func(loss, X, y, weights, omega)
        
    for k in K:
        f = ForwardModeAD(f, Theta_k[k-1])
        
    return f(theta_hat)

'''
Algorithm 3 in Giordano et al. 2019 : Evaluate \delta_w^k\hat{\theta}(w)

'''
def EvaluateDTheta(k, theta_hat, H_inv, Tau_k, Theta_k_minus_1, loss, X, y, weights):
    d = np.zeros(len(theta_hat))
    
    for triple in Tau_k:
        a_i = triple[0]
        K_i = triple[1]
        omega_i = triple[2]
        d = d + a_i * EvaluateTerm(K_i, omega_i, Theta_k_minus_1, theta_hat, loss, X, y, weights)
    return - H_inv @ d


'''
Algorithm 4 in Giordano et al. 2019 : Evaluate \hat{\theta}_{k_{IJ}}^{IJ}(w)

'''
def EvaluateThetaIJ(k_IJ, theta_hat, H_inv, loss, X, y, weights):
    if (k_IJ == 1):
        Tau_star = [[[1, [], 1]]]
    elif (k_IJ == 2):
        Tau_star = [[[1, [], 1]],
                    [[1, [1, 1], 0], [2, [1], 1]]]
    elif (k_IJ == 3):
        Tau_star = [[[1, [], 1]],
                     [[1, [1, 1], 0], [2, [1], 1]],
                     [[3, [1, 2], 0], [3, [2], 1], [3, [1, 1], 1], [1, [1, 1, 1], 0]]]
    else:
        raise Exception("EvaluateThetaIJ currently only built for order k_IJ \in [1, 2, 3]")
        
    Theta_k = []
    t = theta_hat
    for k in range(1, k_IJ + 1):
        Tau_k = Tau_star[k-1]
        partial_w_k_theta = EvaluateDTheta(k, theta_hat, H_inv, Tau_k, Theta_k, loss, X, y, weights)
        Theta_k.append(partial_w_k_theta)
        t = t + (1 / math.factorial(k))*partial_w_k_theta
    return t
