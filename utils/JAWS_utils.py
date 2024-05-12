import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
# np.random.seed(98765)

## Drew added
import tqdm as tqdm
import random
from sklearn import decomposition

from utils import bayesnn ## Code for predictor used in IF experiments

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from datetime import date

## Added for IFs
## Dependencies from RUE

import utils.IF_utils
from utils.IF_utils import *

class InputNormalizer:
    def __init__(self, inputs):
        n = len(inputs)
        self.mean = np.mean(inputs, axis=0)
        _, s, V_T = np.linalg.svd(inputs - self.mean, full_matrices=False)
        self.rotation = V_T.T / s * np.sqrt(n - 1)

    def normalize(self, inputs):
        return np.dot(inputs - self.mean, self.rotation)


class TargetNormalizer:
    def __init__(self, targets):
        self.mean = np.mean(targets)
        self.scale = np.std(targets)

    def normalize(self, targets):
        return (targets - self.mean) / self.scale

    def reverse(self, normalized):
        return self.mean + self.scale * normalized

def sort_both_by_first(v, w):
    zipped_lists = zip(v, w)
    sorted_zipped_lists = sorted(zipped_lists)
    v_sorted = [element for element, _ in sorted_zipped_lists]
    w_sorted = [element for _, element in sorted_zipped_lists]
    
    return [v_sorted, w_sorted]
    

def weighted_quantile(v, w_normalized, q):
    if (len(v) != len(w_normalized)):
        raise ValueError('Error: v is length ' + str(len(v)) + ', but w_normalized is length ' + str(len(w_normalized)))
        
    if (np.sum(w_normalized) > 1.01 or np.sum(w_normalized) < 0.99):
        raise ValueError('Error: w_normalized does not add to 1')
        
    if (q < 0 or 1 < q):
        raise ValueError('Error: Invalid q')

    n = len(v)
    
    v_sorted, w_sorted = sort_both_by_first(v, w_normalized)
    
    cum_w_sum = w_sorted[0]
    i = 0
    while(cum_w_sum <= q):
            i += 1
            cum_w_sum += w_sorted[i]
            
    if (q > 0.5): ## If taking upper quantile: ceil
        return v_sorted[i]
            
    elif (q < 0.5): ## Elif taking lower quantile:
        if (i > 0):
            return v_sorted[i-1]
        else:
            return v_sorted[0]
        
    else: ## Else taking median, return weighted average if don't have cum_w_sum == 0.5
        if (cum_w_sum == 0.5):
            return v_sorted[i]
        
        elif (i > 0):
            return (v_sorted[i]*w_sorted[i] + v_sorted[i-1]*w_sorted[i-1]) / (w_sorted[i] + w_sorted[i-1])
        
        else:
            return v_sorted[0]
        
def logistic_regression_weight_est(X, class_labels):
    clf = LogisticRegression(random_state=0).fit(X, class_labels)
    lr_probs = clf.predict_proba(X)
    return lr_probs[:,1] / lr_probs[:,0]

def random_forest_weight_est(X, class_labels, ntree=100):
    rf = RandomForestClassifier(n_estimators=ntree,criterion='entropy', min_weight_fraction_leaf=0.1).fit(X, class_labels)
    rf_probs = rf.predict_proba(X)
    return rf_probs[:,1] / rf_probs[:,0]

# def get_w(x_pca, x, dataset, bias):
#     if (dataset=='airfoil'):
#         return np.exp(x[:,[0,4]] @ [-bias,bias])
    
#     elif(dataset == 'wine'):
#         return np.exp(x[:,[0,10]] @ [-bias,bias])
    
#     elif(dataset == 'concrete'):
#         return np.exp(np.log(x[:,[0,6]]) @ [-bias,bias])
    
#     ## For communities dataset use top 2 PCs as tilting vars
#     elif (dataset in ['communities', 'naval']):
#         np.random.seed(5)
#         pca = decomposition.PCA(n_components=2)
#         pca.fit(x_pca)
#         x_red = pca.transform(x)
#         return np.exp(x_red @ [-bias,bias])
    
#     ## For communities dataset use top 2 PCs as tilting vars
#     elif (dataset in ['superconduct']):
#         np.random.seed(5)
#         pca = decomposition.PCA(n_components=1)
#         pca.fit(x_pca)
#         x_red = pca.transform(x)
#         return np.exp(x_red @ [bias])
    
#     ## For blog and meps data use logarithm of top 2 PCs as tilting vars
#     else: 
#         np.random.seed(5)
#         pca = decomposition.PCA(n_components=2)
#         pca.fit(x_pca)
#         x_red = pca.transform(x)
#         x_red_min = abs(x_red.min(axis=0))
#         x_red_adjusted = x_red + np.tile(x_red_min + 1, (len(x_red), 1))
#         log_x_red_adjusted = np.log(x_red_adjusted)
#         return np.exp(log_x_red_adjusted @ [-bias,bias])
    
def get_w(x_pca, x, dataset, bias):
#     print("in get_w")
#     print("x    :", x[0:2, :])
#     print("x_pca:", x_pca[0:2, :])
#     print("bias = ", bias)
    
    if (dataset=='airfoil'):
        return np.exp(x[:,[0,4]] @ [-bias,bias])
    
    elif(dataset == 'wine'):
        return np.exp(x[:,[0,10]] @ [-bias,bias])

    
    ## For communities dataset use top 2 PCs as tilting vars
    elif (dataset in ['wave']):
#         np.random.seed(5)
        pca_1 = decomposition.PCA(n_components=1)
        pca_1.fit(x_pca[:, 0:32])
        x_red_1 = pca_1.transform(x[:, 0:32])
        
#         np.random.seed(5)
        pca_2 = decomposition.PCA(n_components=1)
        pca_2.fit(x_pca[:, 32:48])
        x_red_2 = pca_2.transform(x[:, 32:48])
        
        x_red = np.c_[x_red_1, x_red_2]
        
        return np.exp(x_red @ [-bias,bias])
    
        ## For communities dataset use top 2 PCs as tilting vars
    elif (dataset in ['superconduct']):
#         np.random.seed(5)
        pca = decomposition.PCA(n_components=1)
        pca.fit(x_pca)
        x_red = pca.transform(x)
        return np.exp(x_red @ [bias])
    
        ## For communities dataset use top 2 PCs as tilting vars
    elif (dataset in ['communities']):
#         np.random.seed(5)
        pca = decomposition.PCA(n_components=2)
        pca.fit(x_pca)
        x_red = pca.transform(x)
        return np.exp(x_red @ [-bias,bias])
    

def wsample(wts, n, d, frac=0.5):
    n = len(wts) ## n : length or num of weights
    indices = [] ## indices : vector containing indices of the sampled data
    normalized_wts = wts/max(wts)
#     print(normalized_wts[0:20])
    target_num_indices = int(n*frac)
    while(len(indices) < target_num_indices): ## Draw samples until have sampled ~25% of samples from D_test
        proposed_indices = np.where(np.random.uniform(size=n) <= normalized_wts)[0].tolist()
        ## If (set of proposed indices that may add is less than or equal to number still needed): then add all of them
        if (len(proposed_indices) <= target_num_indices - len(indices)):
            for j in proposed_indices:
                indices.append(j)
        else: ## Else: Only add the proposed indices that are needed to get to 25% of D_test
            for j in proposed_indices:
                if(len(indices) < target_num_indices):
                    indices.append(j)
    return(indices)

def exponential_tilting_indices(x_pca, x, dataset, bias=1):
    (n, d) = x.shape
    importance_weights = get_w(x_pca, x, dataset, bias)
#     print("L1 squared : ", np.linalg.norm(weights, ord=1)**2)
#     print("L2 : ", np.linalg.norm(weights, ord=2)**2)
#     print("Effective sample size : ", np.linalg.norm(weights, ord=1)**2 / np.linalg.norm(weights, ord=2)**2)
    return wsample(importance_weights, n, d)

def get_effective_sample_size(X, dataset, bias=1):
    ## Note: x_pca == x
    importance_weights = get_w(X, X, dataset, bias).reshape(len(X))
    sample_size_effective = int(np.linalg.norm(importance_weights, ord=1)**2 / np.linalg.norm(importance_weights, ord=2)**2)
    return sample_size_effective

######################################
# Define 3 regression algorithms
######################################

def leastsq_ridge(X,Y,X1,ridge_mult=0.001):
    lam = ridge_mult * np.linalg.svd(X,compute_uv=False).max()**2
    betahat = np.linalg.solve(\
            np.c_[np.ones(X.shape[0]),X].T.dot(np.c_[np.ones(X.shape[0]),X]) \
                              + lam * np.diag(np.r_[0,np.ones(X.shape[1])]),
            np.c_[np.ones(X.shape[0]),X].T.dot(Y))
    return betahat[0] + X1.dot(betahat[1:])

def random_forest(X,Y,X1,ntree=20):
    rf = RandomForestRegressor(n_estimators=ntree,criterion='mae').fit(X,Y)
    return rf.predict(X1)

def neural_net(X,Y,X1):
    nnet = MLPRegressor(solver='lbfgs',activation='logistic').fit(X,Y)
    return nnet.predict(X1)


def compute_PIs(X,Y,X1,alpha,fit_muh_fun, weights_oracle, dataset, bias, run_effective_sample_size, weights_lr, weights_rf):
    n = len(Y) ## Num training data
    n1 = X1.shape[0] ## Num test data (Note: This is larger than training to focus on confidence estimate)
    
    print("run_effective_sample_size = ", run_effective_sample_size)
    
    if (run_effective_sample_size):
        print("Running effective sample size")
        n_effective = get_effective_sample_size(X, dataset, bias=bias)

    ###############################
    # Naive & jackknife/jack+/jackmm
    ###############################

    muh_vals = fit_muh_fun(X,Y,np.r_[X,X1])
    resids_naive = np.abs(Y-muh_vals[:n])
    muh_vals_testpoint = muh_vals[n:]
    resids_LOO = np.zeros(n)
    muh_LOO_vals_testpoint = np.zeros((n,n1))
    for i in range(n):
        muh_vals_LOO = fit_muh_fun(np.delete(X,i,0),np.delete(Y,i),\
                                   np.r_[X[i].reshape((1,-1)),X1])
        resids_LOO[i] = np.abs(Y[i] - muh_vals_LOO[0])
        muh_LOO_vals_testpoint[i] = muh_vals_LOO[1:]
    ind_q = (np.ceil((1-alpha)*(n+1))).astype(int)
    
    
    ###############################
    # JAW: oracle weights, logistic regression (lr) weights, random forest (rf) weights
    ###############################
    unweighted_upper_vals = (muh_LOO_vals_testpoint.T + resids_LOO).T
    unweighted_lower_vals = (muh_LOO_vals_testpoint.T - resids_LOO).T
    
    ## Add infty (distribution on augmented real line)
    positive_infinity = np.array([float('inf')])
    unweighted_upper_vals = np.vstack((unweighted_upper_vals, positive_infinity*np.ones(n1)))
    unweighted_lower_vals = np.vstack((unweighted_lower_vals, -positive_infinity*np.ones(n1)))
    
    ## Get normalized weights:
    ## [oracle, logistic_regression_estimated, random_forest_estimated]
    weights_full_list = [weights_oracle, weights_lr, weights_rf]
    
    y_upper_JAW_list = [np.zeros(n1), np.zeros(n1), np.zeros(n1)] 
    y_lower_JAW_list = [np.zeros(n1), np.zeros(n1), np.zeros(n1)]
    
    
    for weight_type in range(0, 3):
        weights_full = weights_full_list[weight_type]
        
        weights_normalized = np.zeros((n + 1, n1))
        sum_train_weights = np.sum(weights_full[0:n])
        for j in range(0, n1):
            for i in range(0, n + 1):
                if (i < n):
                    weights_normalized[i, j] = weights_full[i] / (sum_train_weights + weights_full[n + j])
                else:
                    weights_normalized[i, j] = weights_full[n+j] / (sum_train_weights + weights_full[n + j])


        for j in range(0, n1):
            y_upper_JAW_list[weight_type][j] = weighted_quantile(unweighted_upper_vals[:, j], weights_normalized[:, j], 1 - alpha)
            y_lower_JAW_list[weight_type][j] = weighted_quantile(unweighted_lower_vals[:, j], weights_normalized[:, j], alpha)
            
    ###############################
    # JAW-mm: oracle weights, logistic regression (lr) weights, random forest (rf) weights
    ###############################
    
    upper_wq_R_LOO = np.zeros(3)
    
    for weight_type in range(0, 3):
        weights_full = weights_full_list[weight_type]
        weights_normalized = weights_full[0:n] / np.sum(weights_full[0:n])
        upper_wq_R_LOO[weight_type] = weighted_quantile(resids_LOO, weights_normalized, 1 - alpha)
    
    
    ###############################
    # CV+
    ###############################

    K = 10
    n_K = np.floor(n/K).astype(int)
    base_inds_to_delete = np.arange(n_K).astype(int)
    resids_LKO = np.zeros(n)
    muh_LKO_vals_testpoint = np.zeros((n,n1))
    for i in range(K):
        inds_to_delete = (base_inds_to_delete + n_K*i).astype(int)
        muh_vals_LKO = fit_muh_fun(np.delete(X,inds_to_delete,0),np.delete(Y,inds_to_delete),\
                                   np.r_[X[inds_to_delete],X1])
        resids_LKO[inds_to_delete] = np.abs(Y[inds_to_delete] - muh_vals_LKO[:n_K])
        for inner_K in range(n_K):
            muh_LKO_vals_testpoint[inds_to_delete[inner_K]] = muh_vals_LKO[n_K:]
    ind_Kq = (np.ceil((1-alpha)*(n+1))).astype(int)



    ###############################
    # split conformal
    ###############################
    
    idx = np.random.permutation(n)
    n_half = int(np.floor(n/2))
    idx_train, idx_cal = idx[:n_half], idx[n_half:]
    muh_split_vals = fit_muh_fun(X[idx_train],Y[idx_train],np.r_[X[idx_cal],X1])
    resids_split = np.abs(Y[idx_cal]-muh_split_vals[:(n-n_half)])
    muh_split_vals_testpoint = muh_split_vals[(n-n_half):]
    ind_split = (np.ceil((1-alpha)*(n-n_half+1))).astype(int)
   
    ###############################
    # weighted split conformal
    ###############################
    
    ## Add infty (distribution on augmented real line)
    positive_infinity = np.array([float('inf')])
    unweighted_split_vals = np.concatenate([resids_split, positive_infinity])
    
    ## Get normalized weights:
    ## [oracle, logistic_regression_estimated, random_forest_estimated]
    weights_full_list = [weights_oracle, weights_lr, weights_rf]
    
    wsplit_quantiles_list = [np.zeros(n1), np.zeros(n1), np.zeros(n1)]     
    
    for weight_type in range(0, 3):
        weights_full = weights_full_list[weight_type]
        
        weights_normalized_wsplit = np.zeros((n_half + 1, n1))
        sum_cal_weights = np.sum(weights_full[idx_cal])
        for j in range(0, n1):
            for i in range(0, n_half + 1):
                if (i < n_half):
                    i_cal = idx_cal[i]
                    weights_normalized_wsplit[i, j] = weights_full[i_cal] / (sum_cal_weights + weights_full[n + j])
                else:
                    weights_normalized_wsplit[i, j] = weights_full[n+j] / (sum_cal_weights + weights_full[n + j])


        for j in range(0, n1):
            wsplit_quantiles_list[weight_type][j] = weighted_quantile(unweighted_split_vals, weights_normalized_wsplit[:, j], 1 - alpha)
            
    
#     ## Weighted split conformal:
#     weights_split_normalized = weights_full[idx_cal] / np.sum(weights_full[idx_cal])
#     wsplit_quantiles = weighted_quantile(resids_split, weights_split_normalized, 1 - alpha)
    
#     # np.sort(resids_split)[ind_split-1]
    
    
    ###############################
    # Adjustments if doing effective sample size experiments
    ###############################  
    if (run_effective_sample_size):
        ind_q = (np.ceil((1-alpha)*(n_effective+1))).astype(int)
        idx_effective = random.sample(range(0, n), n_effective)
        muh_vals_testpoint = muh_vals_testpoint[idx_effective]
        resids_naive = resids_naive[idx_effective]
        muh_LOO_vals_testpoint = muh_LOO_vals_testpoint[idx_effective, :]
        resids_LOO = resids_LOO[idx_effective]
        
        PIs_dict = {'naive' : pd.DataFrame(\
                    np.c_[muh_vals_testpoint - np.sort(resids_naive)[ind_q-1], \
                        muh_vals_testpoint + np.sort(resids_naive)[ind_q-1]],\
                           columns = ['lower','upper']),\
                'jackknife' : pd.DataFrame(\
                    np.c_[muh_vals_testpoint - np.sort(resids_LOO)[ind_q-1], \
                        muh_vals_testpoint + np.sort(resids_LOO)[ind_q-1]],\
                           columns = ['lower','upper']),\
                'jackknife+' : pd.DataFrame(\
                    np.c_[np.sort(muh_LOO_vals_testpoint.T - resids_LOO,axis=1).T[-ind_q], \
                        np.sort(muh_LOO_vals_testpoint.T + resids_LOO,axis=1).T[ind_q-1]],\
                           columns = ['lower','upper']),\
                'jackknife-mm' : pd.DataFrame(\
                    np.c_[muh_LOO_vals_testpoint.min(0) - np.sort(resids_LOO)[ind_q-1], \
                           muh_LOO_vals_testpoint.max(0) + np.sort(resids_LOO)[ind_q-1]],\
                           columns = ['lower','upper'])}
            

    else:
    ###############################
    # construct prediction intervals
    ###############################
        PIs_dict = {'naive' : pd.DataFrame(\
                    np.c_[muh_vals_testpoint - np.sort(resids_naive)[ind_q-1], \
                        muh_vals_testpoint + np.sort(resids_naive)[ind_q-1]],\
                           columns = ['lower','upper']),\
                'jackknife' : pd.DataFrame(\
                    np.c_[muh_vals_testpoint - np.sort(resids_LOO)[ind_q-1], \
                        muh_vals_testpoint + np.sort(resids_LOO)[ind_q-1]],\
                           columns = ['lower','upper']),\
                'jackknife+' : pd.DataFrame(\
                    np.c_[np.sort(muh_LOO_vals_testpoint.T - resids_LOO,axis=1).T[-ind_q], \
                        np.sort(muh_LOO_vals_testpoint.T + resids_LOO,axis=1).T[ind_q-1]],\
                           columns = ['lower','upper']),\
                'jackknife-mm' : pd.DataFrame(\
                    np.c_[muh_LOO_vals_testpoint.min(0) - np.sort(resids_LOO)[ind_q-1], \
                           muh_LOO_vals_testpoint.max(0) + np.sort(resids_LOO)[ind_q-1]],\
                           columns = ['lower','upper']),\
                'CV+' : pd.DataFrame(\
                    np.c_[np.sort(muh_LKO_vals_testpoint.T - resids_LKO,axis=1).T[-ind_Kq], \
                        np.sort(muh_LKO_vals_testpoint.T + resids_LKO,axis=1).T[ind_Kq-1]],\
                           columns = ['lower','upper']),\
                'split' : pd.DataFrame(\
                    np.c_[muh_split_vals_testpoint - np.sort(resids_split)[ind_split-1], \
                           muh_split_vals_testpoint + np.sort(resids_split)[ind_split-1]],\
                            columns = ['lower','upper']),\
                'weighted_split_oracle' : pd.DataFrame(\
                    np.c_[muh_split_vals_testpoint - wsplit_quantiles_list[0], \
                           muh_split_vals_testpoint + wsplit_quantiles_list[0]],\
                            columns = ['lower','upper']),\
                'weighted_split_lr' : pd.DataFrame(\
                    np.c_[muh_split_vals_testpoint - wsplit_quantiles_list[1], \
                           muh_split_vals_testpoint + wsplit_quantiles_list[1]],\
                            columns = ['lower','upper']),\
                'weighted_split_rf' : pd.DataFrame(\
                    np.c_[muh_split_vals_testpoint - wsplit_quantiles_list[2], \
                           muh_split_vals_testpoint + wsplit_quantiles_list[2]],\
                            columns = ['lower','upper']),\
               'JAW_oracle' : pd.DataFrame(\
                    np.c_[y_lower_JAW_list[0], \
                        y_upper_JAW_list[0]],\
                           columns = ['lower','upper']),\
               'JAW_lr' : pd.DataFrame(\
                    np.c_[y_lower_JAW_list[1], \
                        y_upper_JAW_list[1]],\
                           columns = ['lower','upper']),\
               'JAW_rf' : pd.DataFrame(\
                    np.c_[y_lower_JAW_list[2], \
                        y_upper_JAW_list[2]],\
                           columns = ['lower','upper']),\
                'JAWmm_oracle' : pd.DataFrame(\
                    np.c_[muh_LOO_vals_testpoint.min(0) - upper_wq_R_LOO[0], \
                           muh_LOO_vals_testpoint.max(0) + upper_wq_R_LOO[0]],\
                           columns = ['lower','upper']),\
                'JAWmm_lr' : pd.DataFrame(\
                    np.c_[muh_LOO_vals_testpoint.min(0) - upper_wq_R_LOO[1], \
                           muh_LOO_vals_testpoint.max(0) + upper_wq_R_LOO[1]],\
                           columns = ['lower','upper']),\
                'JAWmm_rf' : pd.DataFrame(\
                    np.c_[muh_LOO_vals_testpoint.min(0) - upper_wq_R_LOO[2], \
                           muh_LOO_vals_testpoint.max(0) + upper_wq_R_LOO[2]],\
                           columns = ['lower','upper'])}
            

        
                
    return pd.concat(PIs_dict.values(), axis=1, keys=PIs_dict.keys())




def compute_PIs_IFs(X,Y,X1,alpha, weights_full, dataset, bias, L2_lambda, itrial, compute_PVs):
    n = len(Y) ## Num training data
    n1 = X1.shape[0] ## Num test data (Note: This is larger than training to focus on confidence estimate)

    
    rng = np.random.RandomState(itrial) ## Generate random state with seed=0
    n_train, n_inputs = X.shape
    n_hidden = 25
    alphas = [np.sqrt(L2_lambda), np.sqrt(L2_lambda)]
    beta = 1.0
    
    print("Dataset = ", dataset)
    
    print("\nTime beginning IFs : ", datetime.now())
    
    ###############################
    # Full model
    ###############################

    model = bayesnn.MLP(n_inputs, n_hidden)
    init_params = model.init_params(rng)

    weights = np.ones(n_train)

    objective, likelihood, prior, likelihood_all = bayesnn.make_objective(model, alphas, beta, n_train, weights)

    config = bayesnn.init_sgd_config()
    config['n_epochs'] = 2000
    config['batch_size'] = 50

    params = bayesnn.train(objective, init_params, X, Y, config, weights)
    muh_vals_testpoint = model.predict(params, X1) ## Same as muh_vals_testpoint

    ###############################
    # Hessian
    ###############################
#     print("likelihood_all", likelihood_all(params, X, Y, weights))
    H = autograd.hessian(objective, 0)(params, X, Y, weights) ## Changed this to objective
#     H_lik = autograd.hessian(likelihood_all, 0)(params, X, Y, weights)
#     eigenvalues_lik, eigenvectors_lik = np.linalg.eig(H_lik)
#     print("mean eigenvalue H_lik: ", np.mean(eigenvalues_lik))
    print("H : ", H)
    max_damp_limit = 10
    damp_search = np.linspace(0.1, max_damp_limit, 10*max_damp_limit+1)

    for damp in damp_search:
        rank = np.linalg.matrix_rank(H + damp * np.eye(len(H)))
        eigenvalues, eigenvectors = np.linalg.eig(H + damp * np.eye(len(H)))
        min_eigenvalue = min(np.abs(eigenvalues))
        if (min_eigenvalue >= 0.5): ## Last had this at 0.01
            break
    print("damp : ", damp)
    
    H = H + damp * np.eye(len(H))
    H_inv = np.linalg.inv(H)
    
    ###############################
    # IF Jackknife+, JAWA, jackknife, and jackknife-mm approximation
    ###############################
    ## 1st order
    resids_LOO_IF1 = np.zeros(n)
    muh_LOO_vals_testpoint_IF1 = np.zeros((n,n1))
    ## 2nd order
    resids_LOO_IF2 = np.zeros(n)
    muh_LOO_vals_testpoint_IF2 = np.zeros((n,n1))
    ## 3rd order
    resids_LOO_IF3 = np.zeros(n)
    muh_LOO_vals_testpoint_IF3 = np.zeros((n,n1))
    for i in range(n):
        weights = np.ones(n_train)
        weights[i] = 0
        params_IFs_1 = EvaluateThetaIJ(1, params, H_inv, objective, X, Y, weights) ## Changed this to objective
        muh_LOO_vals_testpoint_IF1[i, :] = model.predict(params_IFs_1, X1)
        resids_LOO_IF1[i] = np.abs(Y[i] - model.predict(params_IFs_1, X[i:(i+1)]))
        
        params_IFs_2 = EvaluateThetaIJ(2, params, H_inv, objective, X, Y, weights) ## Changed this to objective
        muh_LOO_vals_testpoint_IF2[i, :] = model.predict(params_IFs_2, X1)
        resids_LOO_IF2[i] = np.abs(Y[i] - model.predict(params_IFs_2, X[i:(i+1)]))
        
        params_IFs_3 = EvaluateThetaIJ(3, params, H_inv, objective, X, Y, weights) ## Changed this to objective
        muh_LOO_vals_testpoint_IF3[i, :] = model.predict(params_IFs_3, X1)
        resids_LOO_IF3[i] = np.abs(Y[i] - model.predict(params_IFs_3, X[i:(i+1)]))
        
    ind_q = (np.ceil((1-alpha)*(n+1))).astype(int)
    
    
    
    # Add infinity
    weights_normalized = np.zeros((n + 1, n1))
    sum_train_weights = np.sum(weights_full[0:n])
    for i in range(0, n + 1):
        for j in range(0, n1):
            if (i < n):
                weights_normalized[i, j] = weights_full[i] / (sum_train_weights + weights_full[n + j])
            else:
                weights_normalized[i, j] = weights_full[n+j] / (sum_train_weights + weights_full[n + j])


    unweighted_upper_vals_IF1 = (muh_LOO_vals_testpoint_IF1.T + resids_LOO_IF1).T
    unweighted_lower_vals_IF1 = (muh_LOO_vals_testpoint_IF1.T - resids_LOO_IF1).T
    
    unweighted_upper_vals_IF2 = (muh_LOO_vals_testpoint_IF2.T + resids_LOO_IF2).T
    unweighted_lower_vals_IF2 = (muh_LOO_vals_testpoint_IF2.T - resids_LOO_IF2).T
    
    unweighted_upper_vals_IF3 = (muh_LOO_vals_testpoint_IF3.T + resids_LOO_IF3).T
    unweighted_lower_vals_IF3 = (muh_LOO_vals_testpoint_IF3.T - resids_LOO_IF3).T
    
#     ## Add infty (distribution on augmented real line)
    positive_infinity = np.array([float('inf')])
    unweighted_upper_vals_IF1 = np.vstack((unweighted_upper_vals_IF1, positive_infinity*np.ones(n1)))
    unweighted_lower_vals_IF1 = np.vstack((unweighted_lower_vals_IF1, -positive_infinity*np.ones(n1)))
        
    unweighted_upper_vals_IF2 = np.vstack((unweighted_upper_vals_IF2, positive_infinity*np.ones(n1)))
    unweighted_lower_vals_IF2 = np.vstack((unweighted_lower_vals_IF2, -positive_infinity*np.ones(n1)))
    
    unweighted_upper_vals_IF3 = np.vstack((unweighted_upper_vals_IF3, positive_infinity*np.ones(n1)))
    unweighted_lower_vals_IF3 = np.vstack((unweighted_lower_vals_IF3, -positive_infinity*np.ones(n1)))
        
        
    y_upper_weighted_IF1 = np.zeros(n1)
    y_lower_weighted_IF1 = np.zeros(n1)
    
    y_upper_weighted_IF2 = np.zeros(n1)
    y_lower_weighted_IF2 = np.zeros(n1)
    
    y_upper_weighted_IF3 = np.zeros(n1)
    y_lower_weighted_IF3 = np.zeros(n1)
    
    for j in range(0, n1):
        y_upper_weighted_IF1[j] = weighted_quantile(unweighted_upper_vals_IF1[:, j], weights_normalized[:, j], 1 - alpha)
        y_lower_weighted_IF1[j] = weighted_quantile(unweighted_lower_vals_IF1[:, j], weights_normalized[:, j], alpha)
        
        y_upper_weighted_IF2[j] = weighted_quantile(unweighted_upper_vals_IF2[:, j], weights_normalized[:, j], 1 - alpha)
        y_lower_weighted_IF2[j] = weighted_quantile(unweighted_lower_vals_IF2[:, j], weights_normalized[:, j], alpha)
        
        y_upper_weighted_IF3[j] = weighted_quantile(unweighted_upper_vals_IF3[:, j], weights_normalized[:, j], 1 - alpha)
        y_lower_weighted_IF3[j] = weighted_quantile(unweighted_lower_vals_IF3[:, j], weights_normalized[:, j], alpha)
#         print("y_lower_weighted[j] : ", y_lower_weighted[j])
#         print("y_upper_weighted[j] : ", y_upper_weighted[j])
        
    ###############################
    # JAW-mm
    ###############################
    
    weights_normalized_R_LOO = weights_full[0:n] / np.sum(weights_full[0:n])
    
    upper_wq_R_LOO_IF1 = weighted_quantile(resids_LOO_IF1, weights_normalized_R_LOO, 1 - alpha)
    upper_wq_R_LOO_IF2 = weighted_quantile(resids_LOO_IF2, weights_normalized_R_LOO, 1 - alpha)
    upper_wq_R_LOO_IF3 = weighted_quantile(resids_LOO_IF3, weights_normalized_R_LOO, 1 - alpha)
    
    
    col_names = np.concatenate((['lower' + str(i) for i in range(0, n1)], ['upper' + str(i) for i in range(0, n1)]))
    
    print("\nTime completing IFs : ", datetime.now())
    


    ###############################
    # construct prediction intervals
    ###############################
    if (compute_PVs == False):
        print("\nTime beginning retraining : ", datetime.now())
        for i in range(0, n):
            rng = np.random.RandomState(itrial) ## Generate random state with seed=0
            n_train, n_inputs = X.shape
            n_hidden = 25
            alphas = [np.sqrt(L2_lambda), np.sqrt(L2_lambda)]
            beta = 1.0

            model = bayesnn.MLP(n_inputs, n_hidden)
            init_params = model.init_params(rng)

            weights = np.ones(n_train)
            weights[i] = 0

            objective, likelihood, prior, likelihood_all = bayesnn.make_objective(model, alphas, beta, n_train, weights)

            config = bayesnn.init_sgd_config()
            config['n_epochs'] = 2000
            config['batch_size'] = 50

            params = bayesnn.train(objective, init_params, X, Y, config, weights)
            muh_vals_testpoint = model.predict(params, X1) ## Same as muh_vals_testpoint

            if (i % 20 == 0):

                print("\nTime retraining, i = " + str(i), datetime.now())

        print("\nTime ending retraining : ", datetime.now())
        
    
        PIs_dict = {'IF1-jackknife' : pd.DataFrame(\
                    np.c_[muh_vals_testpoint - np.sort(resids_LOO_IF1)[ind_q-1], \
                        muh_vals_testpoint + np.sort(resids_LOO_IF1)[ind_q-1]],\
                           columns = ['lower','upper']),\
                'IF1-jackknife-mm' : pd.DataFrame(\
                    np.c_[muh_LOO_vals_testpoint_IF1.min(0) - np.sort(resids_LOO_IF1)[ind_q-1], \
                           muh_LOO_vals_testpoint_IF1.max(0) + np.sort(resids_LOO_IF1)[ind_q-1]],\
                           columns = ['lower','upper']),\
                'IF1-jackknife+' : pd.DataFrame(\
                    np.c_[np.sort(muh_LOO_vals_testpoint_IF1.T - resids_LOO_IF1,axis=1).T[-ind_q], \
                        np.sort(muh_LOO_vals_testpoint_IF1.T + resids_LOO_IF1,axis=1).T[ind_q-1]],\
                           columns = ['lower','upper']),\
               'IF1-JAWA' : pd.DataFrame(\
                    np.c_[y_lower_weighted_IF1, \
                        y_upper_weighted_IF1],\
                           columns = ['lower','upper']),\
                'IF1-JAWAmm' : pd.DataFrame(\
                    np.c_[muh_LOO_vals_testpoint_IF1.min(0) - upper_wq_R_LOO_IF1, \
                           muh_LOO_vals_testpoint_IF1.max(0) + upper_wq_R_LOO_IF1],\
                           columns = ['lower','upper']),\
               'IF2-jackknife' : pd.DataFrame(\
                    np.c_[muh_vals_testpoint - np.sort(resids_LOO_IF2)[ind_q-1], \
                        muh_vals_testpoint + np.sort(resids_LOO_IF2)[ind_q-1]],\
                           columns = ['lower','upper']),\
                'IF2-jackknife-mm' : pd.DataFrame(\
                    np.c_[muh_LOO_vals_testpoint_IF2.min(0) - np.sort(resids_LOO_IF2)[ind_q-1], \
                           muh_LOO_vals_testpoint_IF2.max(0) + np.sort(resids_LOO_IF2)[ind_q-1]],\
                           columns = ['lower','upper']),\
                'IF2-jackknife+' : pd.DataFrame(\
                    np.c_[np.sort(muh_LOO_vals_testpoint_IF2.T - resids_LOO_IF2,axis=1).T[-ind_q], \
                        np.sort(muh_LOO_vals_testpoint_IF2.T + resids_LOO_IF2,axis=1).T[ind_q-1]],\
                           columns = ['lower','upper']),\
               'IF2-JAWA' : pd.DataFrame(\
                    np.c_[y_lower_weighted_IF2, \
                        y_upper_weighted_IF2],\
                           columns = ['lower','upper']),\
                'IF2-JAWAmm' : pd.DataFrame(\
                    np.c_[muh_LOO_vals_testpoint_IF2.min(0) - upper_wq_R_LOO_IF2, \
                           muh_LOO_vals_testpoint_IF2.max(0) + upper_wq_R_LOO_IF2],\
                           columns = ['lower','upper']),\
               'IF3-jackknife' : pd.DataFrame(\
                    np.c_[muh_vals_testpoint - np.sort(resids_LOO_IF3)[ind_q-1], \
                        muh_vals_testpoint + np.sort(resids_LOO_IF3)[ind_q-1]],\
                           columns = ['lower','upper']),\
                'IF3-jackknife-mm' : pd.DataFrame(\
                    np.c_[muh_LOO_vals_testpoint_IF3.min(0) - np.sort(resids_LOO_IF3)[ind_q-1], \
                           muh_LOO_vals_testpoint_IF3.max(0) + np.sort(resids_LOO_IF3)[ind_q-1]],\
                           columns = ['lower','upper']),\
                'IF3-jackknife+' : pd.DataFrame(\
                    np.c_[np.sort(muh_LOO_vals_testpoint_IF3.T - resids_LOO_IF3,axis=1).T[-ind_q], \
                        np.sort(muh_LOO_vals_testpoint_IF3.T + resids_LOO_IF3,axis=1).T[ind_q-1]],\
                           columns = ['lower','upper']),\
               'IF3-JAWA' : pd.DataFrame(\
                    np.c_[y_lower_weighted_IF3, \
                        y_upper_weighted_IF3],\
                           columns = ['lower','upper']),\
                'IF3-JAWAmm' : pd.DataFrame(\
                    np.c_[muh_LOO_vals_testpoint_IF3.min(0) - upper_wq_R_LOO_IF3, \
                           muh_LOO_vals_testpoint_IF3.max(0) + upper_wq_R_LOO_IF3],\
                           columns = ['lower','upper'])}
        
        return pd.concat(PIs_dict.values(), axis=1, keys=PIs_dict.keys())
        
        
        
    else: 
        PVs_dict = {'IF1-jackknife' : pd.DataFrame(\
                    np.c_[np.sort(np.tile(muh_vals_testpoint, (n, 1)).T - np.tile(resids_LOO_IF1, (n1, 1))).T, \
                        np.sort(np.tile(muh_vals_testpoint, (n, 1)).T + np.tile(resids_LOO_IF1, (n1, 1))).T],\
                           columns = col_names),\
                'IF1-jackknife+_sorted' : pd.DataFrame(\
                    np.c_[np.sort(muh_LOO_vals_testpoint_IF1.T - resids_LOO_IF1,axis=1).T, \
                        np.sort(muh_LOO_vals_testpoint_IF1.T + resids_LOO_IF1,axis=1).T],\
                           columns = col_names),\
                'IF1-jackknife+_not_sorted' : pd.DataFrame(\
                    np.c_[(muh_LOO_vals_testpoint_IF1.T - resids_LOO_IF1).T, \
                        (muh_LOO_vals_testpoint_IF1.T + resids_LOO_IF1).T],\
                           columns = col_names),\
               'IF2-jackknife' : pd.DataFrame(\
                    np.c_[np.sort(np.tile(muh_vals_testpoint, (n, 1)).T - np.tile(resids_LOO_IF2, (n1, 1))).T, \
                        np.sort(np.tile(muh_vals_testpoint, (n, 1)).T + np.tile(resids_LOO_IF2, (n1, 1))).T],\
                           columns = col_names),\
                'IF2-jackknife+_sorted' : pd.DataFrame(\
                    np.c_[np.sort(muh_LOO_vals_testpoint_IF2.T - resids_LOO_IF2,axis=1).T, \
                        np.sort(muh_LOO_vals_testpoint_IF2.T + resids_LOO_IF2,axis=1).T],\
                           columns = col_names),\
                'IF2-jackknife+_not_sorted' : pd.DataFrame(\
                    np.c_[(muh_LOO_vals_testpoint_IF2.T - resids_LOO_IF2).T, \
                        (muh_LOO_vals_testpoint_IF2.T + resids_LOO_IF2).T],\
                           columns = col_names),\
                'IF3-jackknife' : pd.DataFrame(\
                    np.c_[np.sort(np.tile(muh_vals_testpoint, (n, 1)).T - np.tile(resids_LOO_IF3, (n1, 1))).T, \
                        np.sort(np.tile(muh_vals_testpoint, (n, 1)).T + np.tile(resids_LOO_IF3, (n1, 1))).T],\
                           columns = col_names),\
                'IF3-jackknife+_sorted' : pd.DataFrame(\
                    np.c_[np.sort(muh_LOO_vals_testpoint_IF3.T - resids_LOO_IF3,axis=1).T, \
                        np.sort(muh_LOO_vals_testpoint_IF3.T + resids_LOO_IF3,axis=1).T],\
                           columns = col_names),\
                'IF3-jackknife+_not_sorted' : pd.DataFrame(\
                    np.c_[(muh_LOO_vals_testpoint_IF3.T - resids_LOO_IF3).T, \
                        (muh_LOO_vals_testpoint_IF3.T + resids_LOO_IF3).T],\
                           columns = col_names),\
               'weights_JAW_train' : pd.DataFrame(\
                    np.c_[weights_normalized[0:n, :], \
                        weights_normalized[0:n, :]],\
                           columns = col_names),\
               'weights_JAW_test' : pd.DataFrame(\
                    np.concatenate((weights_normalized[n, :], weights_normalized[n, :])).reshape((1, 2*n1)),\
                           columns = col_names),\
               'muh_vals_testpoint' : pd.DataFrame(\
                    np.concatenate((muh_vals_testpoint, muh_vals_testpoint)).reshape((1, 2*n1)),\
                           columns = col_names)}
    
  
        return pd.concat(PVs_dict.values(), axis=1, keys=PVs_dict.keys())



## Function for obtaining predictive distribution (rather than interval)
def compute_PVs(X,Y,X1,alpha,fit_muh_fun, weights_full, dataset, bias):
    print("Computing predictive distributions")
    n = len(Y) ## Num training data
    n1 = X1.shape[0] ## Num test data (Note: This is larger than training to focus on confidence estimate)
#     n_effective = get_effective_sample_size(X, X, dataset, bias=1)
#     X_effective = X[0:n_effective, :]
#     Y_effective = Y[0:n_effective]

    
    ###############################
    # Naive & jackknife/jack+/jackmm
    ###############################

    muh_vals = fit_muh_fun(X,Y,np.r_[X,X1])
    resids_naive = np.abs(Y-muh_vals[:n])
    muh_vals_testpoint = muh_vals[n:]
    resids_LOO = np.zeros(n)
    muh_LOO_vals_testpoint = np.zeros((n,n1))
    for i in range(n):
        muh_vals_LOO = fit_muh_fun(np.delete(X,i,0),np.delete(Y,i),\
                                   np.r_[X[i].reshape((1,-1)),X1])
        resids_LOO[i] = np.abs(Y[i] - muh_vals_LOO[0])
        muh_LOO_vals_testpoint[i] = muh_vals_LOO[1:]
    ind_q = (np.ceil((1-alpha)*(n+1))).astype(int)
    
    ###############################
    # Naive & jackknife/jack+/jackmm with effective sample size
    ###############################
#     resids_LOO_effective = resids_LOO[0:n_effective]
#     muh_LOO_vals_testpoint_effective = muh_LOO_vals_testpoint[0:n_effective]
#     ind_q_effective = (np.ceil((1-alpha)*(n_effective+1))).astype(int)

    
    
    ###############################
    # Weighted jackknife+
    ###############################
    ## DREW: Double check this later
    
    # Add infinity
    weights_normalized = np.zeros((n + 1, n1))
    sum_train_weights = np.sum(weights_full[0:n])
    for i in range(0, n + 1):
        for j in range(0, n1):
            if (i < n):
                weights_normalized[i, j] = weights_full[i] / (sum_train_weights + weights_full[n + j])
            else:
                weights_normalized[i, j] = weights_full[n+j] / (sum_train_weights + weights_full[n + j])
                

    unweighted_upper_vals = (muh_LOO_vals_testpoint.T + resids_LOO).T
    unweighted_lower_vals = (muh_LOO_vals_testpoint.T - resids_LOO).T
    
#     ## Add infty
    positive_infinity = np.array([float('inf')])
    unweighted_upper_vals = np.vstack((unweighted_upper_vals, positive_infinity*np.ones(n1)))
    unweighted_lower_vals = np.vstack((unweighted_lower_vals, -positive_infinity*np.ones(n1)))
        
    y_upper_weighted = np.zeros(n1)
    y_lower_weighted = np.zeros(n1)
    
    for j in range(0, n1):
        y_upper_weighted[j] = weighted_quantile(unweighted_upper_vals[:, j], weights_normalized[:, j], 1 - alpha)
        y_lower_weighted[j] = weighted_quantile(unweighted_lower_vals[:, j], weights_normalized[:, j], alpha)
        
#     ###############################
#     # CV+
#     ###############################

    K = 10
    n_K = np.floor(n/K).astype(int)
    base_inds_to_delete = np.arange(n_K).astype(int)
    resids_LKO = np.zeros(n)
    muh_LKO_vals_testpoint = np.zeros((n,n1))
    for i in range(K):
        inds_to_delete = (base_inds_to_delete + n_K*i).astype(int)
        muh_vals_LKO = fit_muh_fun(np.delete(X,inds_to_delete,0),np.delete(Y,inds_to_delete),\
                                   np.r_[X[inds_to_delete],X1])
        resids_LKO[inds_to_delete] = np.abs(Y[inds_to_delete] - muh_vals_LKO[:n_K])
        for inner_K in range(n_K):
            muh_LKO_vals_testpoint[inds_to_delete[inner_K]] = muh_vals_LKO[n_K:]
    ind_Kq = (np.ceil((1-alpha)*(n+1))).astype(int)



#     ###############################
#     # split conformal
#     ###############################
    
    idx = np.random.permutation(n)
    n_half = int(np.floor(n/2))
    idx_train, idx_cal = idx[:n_half], idx[n_half:]
    muh_split_vals = fit_muh_fun(X[idx_train],Y[idx_train],np.r_[X[idx_cal],X1])
    resids_split = np.abs(Y[idx_cal]-muh_split_vals[:(n-n_half)])
    muh_split_vals_testpoint = muh_split_vals[(n-n_half):]
    ind_split = (np.ceil((1-alpha)*(n-n_half+1))).astype(int)

    ###############################
    # weighted split conformal
    ###############################
    
    ## Add infty (distribution on augmented real line)
    positive_infinity = np.array([float('inf')])
    unweighted_split_vals = np.concatenate([resids_split, positive_infinity])
    
    ## Get normalized weights:
    
    wsplit_quantiles = np.zeros(n1)   
    
        
    weights_normalized_wsplit = np.zeros((n_half + 1, n1))
    sum_cal_weights = np.sum(weights_full[idx_cal])
    for j in range(0, n1):
        for i in range(0, n_half + 1):
            if (i < n_half):
                i_cal = idx_cal[i]
                weights_normalized_wsplit[i, j] = weights_full[i_cal] / (sum_cal_weights + weights_full[n + j])
            else:
                weights_normalized_wsplit[i, j] = weights_full[n+j] / (sum_cal_weights + weights_full[n + j])


    for j in range(0, n1):
        wsplit_quantiles[j] = weighted_quantile(unweighted_split_vals, weights_normalized_wsplit[:, j], 1 - alpha)
            

    ###############################
    # construct prediction intervals
    ###############################
    
    
    col_names = np.concatenate((['lower' + str(i) for i in range(0, n1)], ['upper' + str(i) for i in range(0, n1)]))
    
#     print(np.c_[np.sort(muh_LOO_vals_testpoint.T - resids_LOO,axis=1).T, \
#                          np.sort(muh_LOO_vals_testpoint.T + resids_LOO,axis=1).T])
    
    PVs_dict = {'naive' : pd.DataFrame(\
                    np.c_[np.sort(np.tile(muh_vals_testpoint, (n, 1)).T - np.tile(resids_naive, (n1, 1)),axis=1).T, \
                        np.sort(np.tile(muh_vals_testpoint, (n, 1)).T + np.tile(resids_naive, (n1, 1)),axis=1).T],\
                           columns = col_names),\
                'jackknife' : pd.DataFrame(\
                    np.c_[np.sort(np.tile(muh_vals_testpoint, (n, 1)).T - np.tile(resids_LOO, (n1, 1))).T, \
                        np.sort(np.tile(muh_vals_testpoint, (n, 1)).T + np.tile(resids_LOO, (n1, 1))).T],\
                           columns = col_names),\
                'jackknife+_sorted' : pd.DataFrame(\
                    np.c_[np.sort(muh_LOO_vals_testpoint.T - resids_LOO,axis=1).T, \
                        np.sort(muh_LOO_vals_testpoint.T + resids_LOO,axis=1).T],\
                           columns = col_names),\
                'jackknife+_not_sorted' : pd.DataFrame(\
                    np.c_[(muh_LOO_vals_testpoint.T - resids_LOO).T, \
                        (muh_LOO_vals_testpoint.T + resids_LOO).T],\
                           columns = col_names),\
                'CV+' : pd.DataFrame(\
                    np.c_[np.sort(muh_LKO_vals_testpoint.T - resids_LKO,axis=1).T, \
                        np.sort(muh_LKO_vals_testpoint.T + resids_LKO,axis=1).T],\
                           columns = col_names),\
                'split' : pd.DataFrame(\
                    np.c_[np.sort(np.tile(muh_split_vals_testpoint, (n_half, 1)).T - np.tile(resids_split, (n1, 1))).T, \
                           np.sort(np.tile(muh_split_vals_testpoint, (n_half, 1)).T + np.tile(resids_split, (n1, 1))).T],\
                            columns = col_names),\
                'split_sorted' : pd.DataFrame(\
                    np.c_[np.sort(np.tile(muh_split_vals_testpoint, (n_half, 1)).T - np.tile(np.sort(resids_split), (n1, 1))).T, \
                           np.sort(np.tile(muh_split_vals_testpoint, (n_half, 1)).T + np.tile(np.sort(resids_split), (n1, 1))).T],\
                            columns = col_names),\
               'weights_split_train' : pd.DataFrame(\
                    np.c_[weights_normalized_wsplit[0:n_half, :], \
                        weights_normalized_wsplit[0:n_half, :]],\
                           columns = col_names),\
               'weights_split_test' : pd.DataFrame(\
                    np.concatenate((weights_normalized_wsplit[n_half, :], weights_normalized_wsplit[n_half, :])).reshape((1, 2*n1)),\
                           columns = col_names),\
               'weights_JAW_train' : pd.DataFrame(\
                    np.c_[weights_normalized[0:n, :], \
                        weights_normalized[0:n, :]],\
                           columns = col_names),\
               'weights_JAW_test' : pd.DataFrame(\
                    np.concatenate((weights_normalized[n, :], weights_normalized[n, :])).reshape((1, 2*n1)),\
                           columns = col_names),\
               'muh_vals_testpoint' : pd.DataFrame(\
                    np.concatenate((muh_vals_testpoint, muh_vals_testpoint)).reshape((1, 2*n1)),\
                           columns = col_names)}
    
                
    return PVs_dict


### Below are 4 functions used for making the AUC error assessment figures in the paper

def prob_interval_sorted_unweighted(PVs_itrial, y_pred_lower, y_pred_upper, test_pt, method):
    ## Find lower point
    idx_low = 0
    train_scores_lower = list(np.sort(PVs_itrial[PVs_itrial['method']==method]['lower' + str(test_pt)]))
    n = len(train_scores_lower)
    while (idx_low < n and train_scores_lower[idx_low] < y_pred_lower):
        idx_low += 1

    ## Find upper point
    idx_up = 0
    train_scores_upper = list(np.sort(PVs_itrial[PVs_itrial['method']==method]['upper' + str(test_pt)]))
    while (idx_up < n and train_scores_upper[idx_up] < y_pred_upper):
        idx_up += 1
    if (idx_up == n or (idx_up > 0 and train_scores_upper[idx_up] >= y_pred_upper)):
        idx_up -= 1
          
    
    lower_quantile = idx_low / (n + 1) ## alpha_E on lower values
    upper_quantile = idx_up / (n + 1) ## 1 - alpha_E on upper values

    
    return min(1 - lower_quantile, upper_quantile)
    
    
    
def prob_interval_JAW(PVs_itrial, y_pred_lower, y_pred_upper, test_pt):
    
    weights = list(PVs_itrial[PVs_itrial['method']=='weights_JAW_train']['lower' + str(test_pt)])
    
    train_scores_lower = list(PVs_itrial[PVs_itrial['method']=='jackknife+_not_sorted']['lower' + str(test_pt)])
    train_scores_upper = list(PVs_itrial[PVs_itrial['method']=='jackknife+_not_sorted']['upper' + str(test_pt)])
    n = len(train_scores_lower) - 1

    ### Add infty
    weights.append(float(PVs_itrial[PVs_itrial['method']=='weights_JAW_test']['lower' + str(test_pt)]))
    positive_infinity = float('inf')
    train_scores_lower.append(-positive_infinity)
    train_scores_upper.append(positive_infinity)

    train_scores_lower_sorted, weights_lower_sorted = sort_both_by_first(train_scores_lower, weights)
    train_scores_upper_sorted, weights_upper_sorted = sort_both_by_first(train_scores_upper, weights)
    

    ## Find lower point
    ## Want low_weight to equal sum of all weights less than a_L + weight of smallest point greater than a_L
    idx_low = 0
    low_weight = weights_lower_sorted[idx_low]
    while (idx_low <= n and train_scores_lower_sorted[idx_low] < y_pred_lower):
        idx_low += 1 
        low_weight += weights_lower_sorted[idx_low]
        
        
        
    ## Find upper point
    idx_up = 0
    up_weight = 0
    while (idx_up <= n and train_scores_upper_sorted[idx_up] < y_pred_upper):
        up_weight += weights_upper_sorted[idx_up]
        idx_up += 1 ## This is id of next one whose weight hasn't been added yet
        
    if (idx_up == n+1 or (idx_up > 0 and train_scores_upper[idx_up] >= y_pred_upper)):
        idx_up -= 1
        up_weight -= weights_upper_sorted[idx_up]
        

    
    beta1 = low_weight ## alpha_E on lower values
    beta2 = up_weight ## 1 - alpha_E on upper values
    
    return min(1 - beta1, beta2)
    
    
def calculate_AUC(PVs_path, dataset, mu, num_taus = 20):
    
    PVs = pd.read_csv(PVs_path)
    
    #### Get tolerance (tau) values based on quantiles on the residuals
    num_trials = np.max(PVs['itrial'])
    y_true = []
    y_preds = []
    for itrial in range(0, np.max(PVs['itrial']) + 1):
        PVs_itrial = PVs[PVs['itrial']==itrial]
        y_true = np.concatenate([y_true, PVs_itrial[PVs_itrial['testpoint']==True][['lower' + str(i) for i in range(0, int(PVs.columns[-1][5:8]) + 1)]].values[0]])
        y_preds = np.concatenate([y_preds, PVs_itrial[PVs_itrial['method']=='muh_vals_testpoint'][['lower' + str(i) for i in range(0, int(PVs.columns[-1][5:8]) + 1)]].values[0]])
    residuals = np.abs(np.array(y_true) - np.array(y_preds))
    taus = np.linspace(np.quantile(residuals, 0.3), np.quantile(residuals, 0.7), num_taus)
    
    
    #### Calculate error probabilities for each method and tau value
    jackknife_plus_probs_ALL = []
    split_probs_ALL = []
    CV_probs_ALL = []
    JAW_probs_ALL = []
    correctness_ALL = []

    for itrial in tqdm.tqdm(set(PVs['itrial'])):
        PVs_itrial = PVs[PVs['itrial']==itrial]
        jackknife_plus_probs_itrial = []
        split_probs_itrial = []
        CV_probs_itrial = []
        JAW_probs_itrial = []
        correctness_itrial = []

        for tau in taus:
            jackknife_plus_probs = []
            split_probs = []
            CV_probs = []
            JAW_probs = []
            correctness = []
            for test_pt in range(0, np.max(PVs['itrial']) + 1):
                y_true = float(PVs_itrial[PVs_itrial['testpoint']==True]['lower' + str(test_pt)])
                y_pred = float(PVs_itrial[PVs_itrial['method']=='muh_vals_testpoint']['lower' + str(test_pt)])
                y_pred_lower = y_pred - tau
                y_pred_upper = y_pred + tau

                jackknife_plus_probs.append(prob_interval_sorted_unweighted(PVs_itrial, y_pred_lower, y_pred_upper, test_pt, 'jackknife+_sorted'))
                split_probs.append(prob_interval_sorted_unweighted(PVs_itrial, y_pred_lower, y_pred_upper, test_pt, 'split'))
                CV_probs.append(prob_interval_sorted_unweighted(PVs_itrial, y_pred_lower, y_pred_upper, test_pt, 'CV+'))
                JAW_probs.append(prob_interval_JAW(PVs_itrial, y_pred_lower, y_pred_upper, test_pt))
                correctness.append(y_pred_lower <= y_true and y_true <= y_pred_upper)

            jackknife_plus_probs_itrial.append(jackknife_plus_probs)
            split_probs_itrial.append(split_probs)
            CV_probs_itrial.append(CV_probs)
            JAW_probs_itrial.append(JAW_probs)
            correctness_itrial.append(correctness)

        jackknife_plus_probs_ALL.append(jackknife_plus_probs_itrial)
        split_probs_ALL.append(split_probs_itrial)
        CV_probs_ALL.append(CV_probs_itrial)
        JAW_probs_ALL.append(JAW_probs_itrial)
        correctness_ALL.append(correctness_itrial)
        
        
    #### Calculate AUC for each method and tau value
    AUCs_jackknife_plus_ALL = []
    AUCs_split_ALL = []
    AUCs_CV_ALL = []
    AUCs_JAW_ALL = []

    for itrial in set(PVs['itrial']):
        AUCs_jackknife_plus_itrial = []
        AUCs_split_itrial = []
        AUCs_CV_itrial = []
        AUCs_JAW_itrial = []

        jackknife_plus_probs_itrial = jackknife_plus_probs_ALL[itrial]
        split_probs_itrial = split_probs_ALL[itrial]
        CV_probs_itrial = CV_probs_ALL[itrial]
        JAW_probs_itrial = JAW_probs_ALL[itrial]
        correctness_itrial = correctness_ALL[itrial]

        for i in range(0, num_taus):
            correctness = correctness_itrial[i]
            AUCs_split_itrial.append(roc_auc_score(correctness, split_probs_itrial[i]))
            AUCs_CV_itrial.append(roc_auc_score(correctness, CV_probs_itrial[i]))
            AUCs_jackknife_plus_itrial.append(roc_auc_score(correctness, jackknife_plus_probs_itrial[i]))
            AUCs_JAW_itrial.append(roc_auc_score(correctness, JAW_probs_itrial[i]))

        AUCs_jackknife_plus_ALL.append(AUCs_jackknife_plus_itrial)
        AUCs_split_ALL.append(AUCs_split_itrial)
        AUCs_CV_ALL.append(AUCs_CV_itrial)
        AUCs_JAW_ALL.append(AUCs_JAW_itrial)

    AUC_results = pd.DataFrame({'taus': taus, 'split': np.mean(AUCs_split_ALL, axis = 0), 
         'jackknife+':np.mean(AUCs_jackknife_plus_ALL, axis = 0),
        'CV+':np.mean(AUCs_CV_ALL, axis = 0),
        'JAW':np.mean(AUCs_JAW_ALL, axis = 0)})
    
    AUC_results_path = './results/JAWE_error_assessment/' + str(date.today()) + '_AUC_' + dataset + '_' + mu + '_' + str(num_trials) + '_Trials.csv'
    AUC_results.to_csv(AUC_results_path)
    
    return AUC_results_path


def plot_AUC(AUC_results_path, dataset, mu, num_trials, save_figure = True):
    AUC_results = pd.read_csv(AUC_results_path)
    display()
    
    plt.figure(figsize=(6,4))
    plt.plot(AUC_results['taus'], AUC_results['JAW'], label = "JAW", color= 'C0', linewidth=4)
    plt.plot(AUC_results['taus'], AUC_results['jackknife+'], label = "Jackknife+", color = 'C2', linewidth=2)
    plt.plot(AUC_results['taus'], AUC_results['CV+'], label = "CV+", color= 'C4', linewidth=2)
    plt.plot(AUC_results['taus'], AUC_results['split'], label = "Split conformal", color = 'C5', linewidth=2)

    plt.ylabel('Average AUROC', fontsize = 16)
    plt.xlabel('Tolerance', fontsize = 16)
    plt.xticks(fontsize= 14)
    plt.yticks(fontsize= 14)
    if (save_figure):
        plt.savefig('./results_figures/fig5and8_ErrorAssessAUC/' + str(date.today()) + '_AUC_' + str(dataset) + '_' + mu + '_' + str(num_trials) + 'Trials.png',dpi=200,bbox_inches='tight')
    plt.show()
        