"""
Classes for full conformal prediction for exchangeable, standard, and feedback covariate shift data,
both for black-box predictive models and computationally optimized for ridge regression, and
functions for (random, exact-coverage) split conformal prediction under standard covariate shift.
Throughout this file, variable name suffixes denote the shape of the numpy array, where
    n: number of training points, or generic number of data points
    n1: n + 1
    p: number of features
    y: number of candidate labels, |Y|
    u: number of sequences in domain, |X|
    m: number of held-out calibration points for split conformal methods
"""

import numpy as np
import time
import scipy as sc

from abc import ABC, abstractmethod

## Drew added
import math
import pandas as pd
import random
from sklearn.neighbors import KernelDensity
from copy import deepcopy
import datetime
import time

# ===== utilities for active learning ptest computation, when using split and adding annotations to calibration =====


def get_f_std(y_std, gpr):
    params = gpr.kernel_.get_params()

    normalized_noise_var = params["k2__noise_level"]
    y_train_var = gpr._y_train_std ** 2

    y_pred_var = y_std ** 2
    f_pred_var = y_pred_var - (y_train_var * normalized_noise_var)
    f_std = np.sqrt(f_pred_var)
    return f_std



def compute_w_ptest_split_active_replacement(cal_test_vals_mat, n_step_curr):
    '''
        @param : cal_test_vals_mat    : matrix (n_step_curr, n_cal + 1). Already normalized for replacement case
        @param : n_step_curr int      : int
        @param : pool_weight_arr_curr : array  (n_step_curr)
    '''
    if (n_step_curr < 1):
        raise ValueError('Error: n_step_curr should be an integer >= 1. Currently, n_step_curr=' + str(n_step_curr))
      
    #if ((len(cal_test_vals_mat) != n_step_curr) or (len(pool_weight_arr_curr) != n_step_curr)):
    #     raise ValueError('Error: should have len(cal_test_vals_mat) == n_step_curr == len(pool_weight_arr_curr)')
                
    if (n_step_curr == 1):
        ## If currently one step past IID
        return cal_test_vals_mat[-1]
        
    n_cal_test = np.shape(cal_test_vals_mat)[1]
    adjusted_vals = deepcopy(cal_test_vals_mat[-1])
    idx_include = np.repeat(True, n_cal_test)
    
    if (len(adjusted_vals) != len(idx_include)):
        raise ValueError('Error: should have len(adjusted_vals) == len(idx_include)')
      

    for i in range(n_cal_test):
        idx_include[i] = False
        idx_include[i-1] = True
        total = compute_w_ptest_split_active_replacement_helper(cal_test_vals_mat[:-1,idx_include], n_step_curr-1)
        adjusted_vals[i] = adjusted_vals[i] * total
    return adjusted_vals
            
        
def compute_w_ptest_split_active_replacement_helper(cal_test_vals_mat, n_step_curr):
    if (n_step_curr == 1):
#         print(np.sum(cal_test_vals_mat / (pool_weight_arr_curr[-1] + cal_test_vals_mat)))
        return np.sum(cal_test_vals_mat)
    
    else:
        total = 0
        n_cal_test = np.shape(cal_test_vals_mat)[1]
        idx_include = np.repeat(True, n_cal_test)
        for i in range(n_cal_test):
            idx_include[i] = False
            idx_include[i-1] = True
            total += cal_test_vals_mat[-1,i]*compute_w_ptest_split_active_replacement_helper(cal_test_vals_mat[:-1,idx_include], n_step_curr - 1) 
        return total




def compute_weight_bound(beta, exp_vars_cal, exp_vars_pool_sorted_descending, alpha=0.1):
    """
    :param beta          : Largest acceptable probability of infinite interval widths
    :param exp_vars_cal  : Unnormalized, unbounded weights (exponentiated variances for calibration data)
    :param exp_vars_pool_sorted_descending : Unnormalized, unbounded weights (exponentiated variances for pool) already sorted in descending order
    Compute B : largest bound for which the probability of an infinite interval width is maintained below beta
    """
    
    for i, b in enumerate(exp_vars_pool_sorted_descending):
        
        total_weight_cal = np.sum(np.minimum(exp_vars_cal, b))
        unnormalized_weights_pool = np.minimum(exp_vars_pool_sorted_descending, b)
        inds_infty_pool = np.where(unnormalized_weights_pool / (total_weight_cal + unnormalized_weights_pool) >= alpha)[0]
                
        if (np.sum(unnormalized_weights_pool[inds_infty_pool]) / np.sum(unnormalized_weights_pool) <= beta):
            print("iterated through ", i , " out of ", len(exp_vars_pool_sorted_descending))
            break
    
    return b




def compute_weight_bound_binary_search(beta, exp_vars_cal, exp_vars_pool_sorted_descending, start, end, alpha=0.1):
    """
    :param beta          : Largest acceptable probability of infinite interval widths
    :param exp_vars_cal  : Unnormalized, unbounded weights (exponentiated variances for calibration data)
    :param exp_vars_pool_sorted_descending : Unnormalized, unbounded weights (exponentiated variances for pool) already sorted in descending order
    Compute B : largest bound for which the probability of an infinite interval width is maintained below beta
    """
    mid = int((end + start) / 2)
    
    # print("start : ", start, "mid : ", mid, "end : ", end)
    
    if (start + 1 == end):
        return exp_vars_pool_sorted_descending[start]
    
    
    b = exp_vars_pool_sorted_descending[mid]
    
    total_weight_cal = np.sum(np.minimum(exp_vars_cal, b))
    unnormalized_weights_pool = np.minimum(exp_vars_pool_sorted_descending, b)
    inds_infty_pool = np.where(unnormalized_weights_pool / (total_weight_cal + unnormalized_weights_pool) >= alpha)[0]
                
    if (np.sum(unnormalized_weights_pool[inds_infty_pool]) / np.sum(unnormalized_weights_pool) <= beta):

        return compute_weight_bound_binary_search(beta, exp_vars_cal, exp_vars_pool_sorted_descending, start=mid, end=end, alpha=0.1)

    else:
        return compute_weight_bound_binary_search(beta, exp_vars_cal, exp_vars_pool_sorted_descending, start=start, end=mid, alpha=0.1)
            
    

    
    
# Small adjustment of cal_test_vals_mat[:-1,i] instead of cal_test_vals_mat[-1,i]
def compute_w_ptest_split_active_no_replacement(cal_test_vals_mat, n_step_curr, pool_weight_arr_curr):
    '''
        @param : cal_test_vals_mat    : matrix (n_step_curr, n_cal + 1)
        @param : n_step_curr int      : int
        @param : pool_weight_arr_curr : array  (n_step_curr)
    '''
    if (n_step_curr < 1):
        raise ValueError('Error: n_step_curr should be an integer >= 1. Currently, n_step_curr=' + str(n_step_curr))
      
    #if ((len(cal_test_vals_mat) != n_step_curr) or (len(pool_weight_arr_curr) != n_step_curr)):
    #     raise ValueError('Error: should have len(cal_test_vals_mat) == n_step_curr == len(pool_weight_arr_curr)')
                
    if (n_step_curr == 1):
        ## If currently one step past IID
        ## Here, cal_test_vals_mat will just be an array
        return cal_test_vals_mat / (pool_weight_arr_curr[-1] + cal_test_vals_mat)
        
    n_cal_test = np.shape(cal_test_vals_mat)[1]
    adjusted_vals = deepcopy(cal_test_vals_mat[-1] / (pool_weight_arr_curr[-1] + cal_test_vals_mat[-1]))
    idx_include = np.repeat(True, n_cal_test)
    
    if (len(adjusted_vals) != len(idx_include)):
        raise ValueError('Error: should have len(adjusted_vals) == len(idx_include)')
      

    for i in range(n_cal_test):
        idx_include[i] = False
        idx_include[i-1] = True
        total = compute_w_ptest_split_active_no_replacement_helper(cal_test_vals_mat[:-1,idx_include], n_step_curr-1, pool_weight_arr_curr[:-1] + cal_test_vals_mat[:-1,i])
        adjusted_vals[i] = adjusted_vals[i] * total
    return adjusted_vals
            
        
def compute_w_ptest_split_active_no_replacement_helper(cal_test_vals_mat, n_step_curr, pool_weight_arr_curr):
    if (n_step_curr == 1):
#         print(np.sum(cal_test_vals_mat / (pool_weight_arr_curr[-1] + cal_test_vals_mat)))
        return np.sum(cal_test_vals_mat / (pool_weight_arr_curr[-1] + cal_test_vals_mat))
    
    else:
        total = 0
        n_cal_test = np.shape(cal_test_vals_mat)[1]
        idx_include = np.repeat(True, n_cal_test)
        for i in range(n_cal_test):
            idx_include[i] = False
            idx_include[i-1] = True
            total += (cal_test_vals_mat[-1,i] / (pool_weight_arr_curr[-1] + cal_test_vals_mat[-1,i]))*compute_w_ptest_split_active_no_replacement_helper(cal_test_vals_mat[:-1,idx_include], n_step_curr - 1, pool_weight_arr_curr[:-1] + cal_test_vals_mat[:-1,i]) 
        return total



            
def weight_adjustment(x):
    out = np.zeros(len(x))
    for i in range(0, len(x)):
        out[i] += x[i]*(np.sum(x[:i]) + np.sum(x[i+1:]))
        
    return out


# ===== utilities for KDE density estimation =====

def KDE_density_estimates(X, bandwidth=0.5):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
    density = np.exp(kde.score_samples(X))
    return density / np.sum(density)

## Compute test density for active learning experiments
def std_to_test_density(std_vals):
    var_vals = std_vals**2
    return var_vals / np.sum(var_vals)

# ===== utilities for split conformal =====

def get_split_coverage(lu_nx2, y_n):
    """
    Computes empirical coverage of split conformal confidence interval
    :param lu_nx2: (n, 2) numpy array where first and second columns are lower and upper endpoints
    :param y_n: (n,) numpy array of true labels
    :return: float, empirical coverage
    """
    cov = np.sum((y_n >= lu_nx2[:, 0]) & (y_n <= lu_nx2[:, 1])) / y_n.size
    return cov

def get_randomized_staircase_coverage(C_n, y_n):
    """
    Computes empirical coverage and lengths of randomized staircase confidence sets.

    :param C_n: length-n list of outputs of get_randomized_staircase_confidence_set (i.e., list of tuples)
    :param y_n: (n,) numpy array of true labels
    :return: (n,) binary array of coverage and (n,) numpy array of lengths
    """
    def is_covered(confint_list, y):
        for confint_2 in confint_list:
            if y >= confint_2[0] and y <= confint_2[1]:
                return True
        return False
    def get_len_conf_set(confint_list):
        return np.sum([confint_2[1] - confint_2[0] for confint_2 in confint_list])

    cov_n = np.array([is_covered(confset, y) for confset, y in zip(C_n, y_n)])
    len_n = np.array([get_len_conf_set(confset) for confset in C_n])
    return cov_n, len_n

def get_randomized_staircase_confidence_set(scores_m, weights_m1, predtest, alpha: float = 0.1):
    """
    Computes the "randomized staircase" confidence set in Alg. S1.

    :param scores_m: (m,) numpy array of calibration scores
    :param weights_m1: (m + 1) numpy array of calibration weights and single test weight
    :param predtest: float, prediction on test input
    :param alpha: miscoverage level
    :return: list of tuples (l, u), where l and u are floats denoting lower and upper
        endpoints of an interval.
    """
    lb_is_set = False
    idx = np.argsort(scores_m)
    sortedscores_m1 = np.hstack([0, scores_m[idx]])
    sortedweights_m1 = np.hstack([0, weights_m1[: -1][idx]])
    C = []

    # interval that is deterministically included in the confidence set
    # (color-coded green in Fig. S1)
    cdf_m1 = np.cumsum(sortedweights_m1) # CDF up to i-th sorted calibration score
    cdf_plus_test_weight_m1 = cdf_m1 + weights_m1[-1]
    deterministic_idx = np.where(cdf_plus_test_weight_m1 < 1 - alpha)[0]
    if deterministic_idx.size:
        i_det = np.max(deterministic_idx)
        C.append((predtest - sortedscores_m1[i_det + 1], predtest + sortedscores_m1[i_det + 1]))

    # intervals that are randomly included in the confidence set
    # (color-coded teal and blue in Fig. S1)
    for i in range(i_det + 1, sortedscores_m1.size - 1):
        assert(cdf_plus_test_weight_m1[i] >= 1 - alpha)
        if cdf_plus_test_weight_m1[i] >= 1 - alpha and cdf_m1[i] < 1 - alpha:
            if not lb_is_set:
                lb_is_set = True
                LF = cdf_m1[i]
            F = (cdf_plus_test_weight_m1[i] - (1 - alpha)) / (cdf_m1[i] + weights_m1[-1] - LF)
            if sc.stats.bernoulli.rvs(1 - F):
                C.append((predtest + sortedscores_m1[i], predtest + sortedscores_m1[i + 1]))
                C.append((predtest - sortedscores_m1[i + 1], predtest - sortedscores_m1[i]))

    # halfspaces that are randomly included in the confidence set
    # (color-coded purple in Fig. S1)
    if cdf_m1[-1] < 1 - alpha:  # sum of all calibration weights
        if not lb_is_set:
            LF = cdf_m1[-1]
        F = alpha / (1 - LF)
        if sc.stats.bernoulli.rvs(1 - F):
            C.append((predtest + sortedscores_m1[-1], np.inf))
            C.append((-np.inf, predtest - sortedscores_m1[-1]))
    return C



# ========== full conformal utilities ==========

def get_weighted_quantile(quantile, w_n1xy, scores_n1xy):
    """
    Compute the quantile of weighted scores for each candidate label y

    :param quantile: float, quantile
    :param w_n1xy: (n + 1, |Y|) numpy array of weights (unnormalized)
    :param scores_n1xy: (n + 1, |Y|) numpy array of scores
    :return: (|Y|,) numpy array of quantiles
    """
    if w_n1xy.ndim == 1:
        w_n1xy = w_n1xy[:, None]
        scores_n1xy = scores_n1xy[:, None]

    # normalize probabilities
    p_n1xy = w_n1xy / np.sum(w_n1xy, axis=0)

    # sort scores and their weights accordingly
    sorter_per_y_n1xy = np.argsort(scores_n1xy, axis=0)
    sortedscores_n1xy = np.take_along_axis(scores_n1xy, sorter_per_y_n1xy, axis=0)
    sortedp_n1xy = np.take_along_axis(p_n1xy, sorter_per_y_n1xy, axis=0)

    # locate quantiles of weighted scores per y
    cdf_n1xy = np.cumsum(sortedp_n1xy, axis=0)
    qidx_y = np.sum(cdf_n1xy < quantile, axis=0)  # equivalent to [np.searchsorted(cdf_n1, q) for cdf_n1 in cdf_n1xy]
    q_y = sortedscores_n1xy[(qidx_y, range(qidx_y.size))]
    return q_y

def is_covered(y, confset, y_increment):
    """
    Return if confidence set covers true label

    :param y: true label
    :param confset: numpy array of values in confidence set
    :param y_increment: float, \Delta increment between candidate label values, 0.01 in main paper
    :return: bool
    """
    return np.any(np.abs(y - confset) < (y_increment / 2))

# ========== JAW utilities ==========

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
#         print(np.sum(w_normalized))
#         print(w_normalized)
        raise ValueError('Error: w_normalized does not add to 1')
        
    if (q < 0 or 1 < q):
        raise ValueError('Error: Invalid q')

    n = len(v)
    
    v_sorted, w_sorted = sort_both_by_first(v, w_normalized)
    
    w_sorted_cum = np.cumsum(w_sorted)
    
#     cum_w_sum = w_sorted[0]
    i = 0
    while(w_sorted_cum[i] < q):
        i += 1
#         cum_w_sum += w_sorted[i]
        
    
            
    if (q > 0.5): ## If taking upper quantile: ceil
#         print("w_sorted_cum[i]",i, v_sorted[i], w_sorted_cum[i])
        return v_sorted[i]
            
    elif (q < 0.5): ## Elif taking lower quantile:
        
        if (i > 0 and w_sorted_cum[i] == q):
            return v_sorted[i]
        elif (i > 0):
#             print("w_sorted_cum[i-1]",i-1, v_sorted[i-1], w_sorted_cum[i-1])
            return v_sorted[i-1]
        else:
            return v_sorted[0]
        
    else: ## Else taking median, return weighted average if don't have cum_w_sum == 0.5
        if (w_sorted_cum[i] == 0.5):
            return v_sorted[i]
        
        elif (i > 0):
            return (v_sorted[i]*w_sorted[i] + v_sorted[i-1]*w_sorted[i-1]) / (w_sorted[i] + w_sorted[i-1])
        
        else:
            return v_sorted[0]

        
# ========== utilities and classes for weight estimation with probabilistic classification ==========
        
def logistic_regression_weight_est(X, class_labels):
    clf = LogisticRegression(random_state=0).fit(X, class_labels)
    lr_probs = clf.predict_proba(X)
    return lr_probs[:,1] / lr_probs[:,0]

def random_forest_weight_est(X, class_labels, ntree=100):
    rf = RandomForestClassifier(n_estimators=ntree,criterion='entropy', min_weight_fraction_leaf=0.1).fit(X, class_labels)
    rf_probs = rf.predict_proba(X)
    return rf_probs[:,1] / rf_probs[:,0]


# ========== utilities and classes for full conformal with ridge regression ==========

def get_invcov_dot_xt(X_nxp, gamma, use_lapack: bool = True):
    """
    Compute (X^TX + \gamma I)^{-1} X^T

    :param X_nxp: (n, p) numpy array encoding sequences
    :param gamma: float, ridge regularization strength
    :param use_lapack: bool, whether or not to use low-level LAPACK functions for inverting covariance (fastest)
    :return: (p, n) numpy array, (X^TX + \gamma I)^{-1} X^T
    """
    reg_pxp = gamma * np.eye(X_nxp.shape[1])
    reg_pxp[0, 0] = 0  # don't penalize intercept term
    cov_pxp = X_nxp.T.dot(X_nxp) + reg_pxp
    if use_lapack:
        # fastest way to invert PD matrices from
        # https://stackoverflow.com/questions/40703042/more-efficient-way-to-invert-a-matrix-knowing-it-is-symmetric-and-positive-semi
        zz, _ = sc.linalg.lapack.dpotrf(cov_pxp, False, False)
        invcovtri_pxp, info = sc.linalg.lapack.dpotri(zz)
        assert(info == 0)
        invcov_pxp = np.triu(invcovtri_pxp) + np.triu(invcovtri_pxp, k=1).T
    else:
        invcov_pxp = sc.linalg.pinvh(cov_pxp)
    return invcov_pxp.dot(X_nxp.T)


class ConformalRidge(ABC):
    """
    Abstract base class for full conformal with computations optimized for ridge regression.
    """
    def __init__(self, ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack: bool = True):
        """
        :param ptrain_fn: function that outputs likelihood of input under training input distribution, p_X
        :param ys: numpy array of candidate labels
        :param Xuniv_uxp: (u, p) numpy array encoding all sequences in domain (e.g., all 2^13 sequences
            in Poelwijk et al. 2019 data set), needed for computing normalizing constant
        :param gamma: float, ridge regularization strength
        :param use_lapack: bool, whether or not to use low-level LAPACK functions for inverting covariance (fastest)
        """
        self.ptrain_fn = ptrain_fn
        self.Xuniv_uxp = Xuniv_uxp
        self.p = Xuniv_uxp.shape[1]
        self.ys = ys
        self.n_y = ys.size
        self.gamma = gamma
        self.use_lapack = use_lapack

    def get_normalizing_constant(self, beta_p, lmbda):
        predall_u = self.Xuniv_uxp.dot(beta_p)
        Z = np.sum(np.exp(lmbda * predall_u))
        return Z

    def get_insample_scores(self, Xaug_n1xp, ytrain_n):
        """
        Compute in-sample scores, i.e. residuals using model trained on all n + 1 data points (instead of LOO data)

        :param Xaug_n1xp: (n + 1, p) numpy array encoding all n + 1 sequences (training + candidate test point)
        :param ytrain_n: (n,) numpy array of true labels for the n training points
        :return: (n + 1, |Y|) numpy array of scores
        """
        A = get_invcov_dot_xt(Xaug_n1xp, self.gamma, use_lapack=self.use_lapack)
        C = A[:, : -1].dot(ytrain_n)  # p elements
        a_n1 = C.dot(Xaug_n1xp.T)
        b_n1 = A[:, -1].dot(Xaug_n1xp.T)

        # process in-sample scores for each candidate value y
        scoresis_n1xy = np.zeros([ytrain_n.size + 1, self.n_y])
        by_n1xy = np.outer(b_n1, self.ys)
        muhatiy_n1xy = a_n1[:, None] + by_n1xy
        scoresis_n1xy[: -1] = np.abs(ytrain_n[:, None] - muhatiy_n1xy[: -1])
        scoresis_n1xy[-1] = np.abs(self.ys - muhatiy_n1xy[-1])
        return scoresis_n1xy

    
    

    def compute_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda, compute_lrs: bool = True, n_step_curr=1):
#         timestamp = time.time()
#         value = datetime.datetime.fromtimestamp(timestamp)
#         print("time=", value.strftime('%Y-%m-%d %H:%M:%S'))
#         print("inside compute_loo_scores_and_lrs. time=", time.time())
#         print("compute_lrs", compute_lrs)
#         print("n_step_curr", n_step_curr)
        """
        Compute LOO scores, i.e. residuals using model trained on n data points (training + candidate test points,
        but leave i-th training point out).

        :param Xaug_n1xp: (n + 1, p) numpy array encoding all n + 1 sequences (training + candidate test point)
        :param ytrain_n: (n,) numpy array of true labels for the n training points
        :param lmbda: float, inverse temperature of design algorithm in Eq. 6, {0, 2, 4, 6} in main paper
        :param compute_lrs: bool: whether or not to compute likelihood ratios (this part takes the longest,
            so set to False if only want to compute scores)
        :return: (n + 1, |Y|) numpy arrays of scores S_i(X_test, y) and weights w_i^y(X_test) in Eq. 3 in main paper
        """
        if (n_step_curr < 1):
            raise ValueError('Error: n_step_curr should be an integer >= 1. Currently, n_step_curr=' + str(n_step_curr))


        ## Call subroutine for computing one-step full CP LOO scores.
        scoresloo_n1xy, w_n1xy = self.compute_onestep_full_cp_loo_scores_weights_ridge(Xaug_n1xp, ytrain_n, lmbda, compute_lrs)
        
#         print("w_n1xy shape", np.shape(w_n1xy))
        
        # likelihood ratios for each candidate value y
        w_n1xy_adjusted = None
        if compute_lrs:

            if (n_step_curr == 1):
                ## If currently one step past IID, then return the scores and one-step weights
                return scoresloo_n1xy, w_n1xy

            w_n1xy_adjusted = deepcopy(w_n1xy)
            idx_include_y = np.repeat(True, len(ytrain_n))
            idx_include_x = np.repeat(True, len(ytrain_n)+1)
            
            ## For each column in matrix
            for i in range(len(ytrain_n)):
                idx_include_y[i]   = False
                idx_include_y[i-1] = True
                idx_include_x[i]   = False
                idx_include_x[i-1] = True
                total = self.compute_mfcs_full_cp_scores_weights_ridge_helper(Xaug_n1xp[idx_include_x], ytrain_n[idx_include_y], lmbda, n_step_curr-1)
                w_n1xy_adjusted[i] = w_n1xy_adjusted[i] * total
   
            ## Weight adjustment for test point
            total = self.compute_mfcs_full_cp_scores_weights_ridge_helper(Xaug_n1xp[:-1], ytrain_n, lmbda, n_step_curr-1, False)
            w_n1xy_adjusted[-1] = w_n1xy_adjusted[-1] * total
        
        return scoresloo_n1xy, w_n1xy_adjusted

    
    
    @abstractmethod
    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda, n_step_curr):
        pass
    
    
    
    
    def compute_mfcs_full_cp_scores_weights_ridge_helper(self, Xaug_n1xp_, ytrain_n_, lmbda, n_step_curr, includes_test: bool = True):
        ## Call subroutine for computing full CP LOO scores
#         print("Xaug_n1xp_ shape : ", np.shape(Xaug_n1xp_))
#         print("ytrain_n_ shape : ", np.shape(ytrain_n_))
        
        scoresloo_n1xy_, w_n1xy_ = self.compute_onestep_full_cp_loo_scores_weights_ridge(Xaug_n1xp_, ytrain_n_, lmbda, True, includes_test)
#         print("w_n1xy_ shape : ", np.shape(w_n1xy_))
        
        if (n_step_curr == 1):
            return np.sum(w_n1xy_, axis=0)

        else:
            total = np.repeat(0.0, self.n_y) ## len(total) = number of y candidates
            idx_include_y = np.repeat(True, len(ytrain_n_)) ## len(idx_include_y) = number of training points
            idx_include_x = np.repeat(True, len(Xaug_n1xp_)) ## len(idx_include_x) = number of training points + 1 test point, if includes_test==True
            ## For each row (datapoint), add to the total summation the weights for that point * recursive summation on remaining points
            
#             print("Xaug_n1xp_[idx_include_x] shape : ", np.shape(Xaug_n1xp_[idx_include_x]))
#             print("ytrain_n_[idx_include_y] shape : ", np.shape(ytrain_n_[idx_include_y]))
            
            for i in range(len(w_n1xy_)):
                if ((i < len(w_n1xy_)-1) or (includes_test == False)):
                    idx_include_y[i]   = False
                    idx_include_y[i-1] = True
                    idx_include_x[i]   = False
                    idx_include_x[i-1] = True
                    total += w_n1xy_[i]*self.compute_mfcs_full_cp_scores_weights_ridge_helper(Xaug_n1xp_[idx_include_x], ytrain_n_[idx_include_y], lmbda, n_step_curr - 1, includes_test)
                else:
#                     print("else condition in helper")
                    ## Else == (i==len(w_n1xy_)-1 and includes_test == True): 
                        ## Then, removing i is removing the test point, so recurse with includes_test=False
                    total += w_n1xy_[i]*self.compute_mfcs_full_cp_scores_weights_ridge_helper(Xaug_n1xp_[:-1], ytrain_n_, lmbda, n_step_curr - 1, False)
#             print("final i", i)
#             print("len(w_n1xy_)", len(w_n1xy_))
            
            return total



    def compute_onestep_full_cp_loo_scores_weights_ridge(self, Xaug_n1xp, ytrain_n, lmbda, compute_lrs: bool = True, includes_test: bool = True):
        
        if (includes_test == True):
            ## This condition is for default case where the last row is the test point ()
            # fit n + 1 LOO models and store linear parameterizations of \mu_{-i, y}(X_i) as function of y
            n = ytrain_n.size
            ab_nx2 = np.zeros([n, 2])
            C_nxp = np.zeros([n, self.p])
            An_nxp = np.zeros([n, self.p])
            for i in range(n):
                # construct A_{-i}
                Xi_nxp = np.vstack([Xaug_n1xp[: i], Xaug_n1xp[i + 1 :]]) # n rows
                Ai = get_invcov_dot_xt(Xi_nxp, self.gamma, use_lapack=self.use_lapack)

                # compute linear parameterizations of \mu_{-i, y}(X_i)
                yi_ = np.hstack([ytrain_n[: i], ytrain_n[i + 1 :]])  # n - 1 elements
                Ci = Ai[:, : -1].dot(yi_) # p elements
                ai = Ci.dot(Xaug_n1xp[i])  # = Xtrain_nxp[i]
                bi = Ai[:, -1].dot(Xaug_n1xp[i])

                # store
                ab_nx2[i] = ai, bi
                C_nxp[i] = Ci
                An_nxp[i] = Ai[:, -1]

            # LOO score for i = n + 1
            tmp = get_invcov_dot_xt(Xaug_n1xp[: -1], self.gamma, use_lapack=self.use_lapack)
            beta_p = tmp.dot(ytrain_n)
            alast = beta_p.dot(Xaug_n1xp[-1])  # prediction a_{n + 1}. Xaug_n1xp[-1] = Xtest_p

            # process LOO scores for each candidate value y
            scoresloo_n1xy = np.zeros([n + 1, self.n_y])
            by_nxy = np.outer(ab_nx2[:, 1], self.ys)
            prediy_nxy = ab_nx2[:, 0][:, None] + by_nxy
            scoresloo_n1xy[: -1] = np.abs(ytrain_n[:, None] - prediy_nxy)
            scoresloo_n1xy[-1] = np.abs(self.ys - alast)

            ## Compute one-step weights w_n1xy
            w_n1xy = None
            if compute_lrs:
                
                betaiy_nxpxy = C_nxp[:, :, None] + self.ys * An_nxp[:, :, None]
                # compute normalizing constant in Eq. 6 in main paper
                pred_nxyxu = np.tensordot(betaiy_nxpxy, self.Xuniv_uxp, axes=(1, 1))
                normconst_nxy = np.sum(np.exp(lmbda * pred_nxyxu), axis=2)
                ptrain_n = self.ptrain_fn(Xaug_n1xp[: -1])

                ## Each row of these are the n+1 "one-step" weights for a specific candidate label
                w_n1xy = np.zeros([n + 1, self.n_y])
                wi_num_nxy = np.exp(lmbda * prediy_nxy)
                w_n1xy[: -1] = wi_num_nxy / (ptrain_n[:, None] * normconst_nxy)

                # for last i = n + 1, which is constant across candidate values of y
                Z = self.get_normalizing_constant(beta_p, lmbda)
                w_n1xy[-1] = np.exp(lmbda * alast) / (self.ptrain_fn(Xaug_n1xp[-1][None, :]) * Z)

            return scoresloo_n1xy, w_n1xy
        
        else:
            ## This condition is for when every row is a test point, so len(Xaug_n1xp) = len(ytrain_n)
            # fit n + 1 LOO models and store linear parameterizations of \mu_{-i, y}(X_i) as function of y
            
            
#             A = get_invcov_dot_xt(Xaug_n1xp, self.gamma, use_lapack=self.use_lapack)
#             C = A[:, : -1].dot(ytrain_n)  # p elements
#             a_n1 = C.dot(Xaug_n1xp.T)
#             b_n1 = A[:, -1].dot(Xaug_n1xp.T)

#             # process in-sample scores for each candidate value y
#             scoresis_n1xy = np.zeros([ytrain_n.size + 1, self.n_y])
#             by_n1xy = np.outer(b_n1, self.ys)
#             muhatiy_n1xy = a_n1[:, None] + by_n1xy
#             scoresis_n1xy[: -1] = np.abs(ytrain_n[:, None] - muhatiy_n1xy[: -1])
#             scoresis_n1xy[-1] = np.abs(self.ys - muhatiy_n1xy[-1])
#             return scoresis_n1xy
            
            
            n = ytrain_n.size
            ab_nx1 = np.zeros([n, 1])
            C_nxp = np.zeros([n, self.p])
#             An_nxp = np.zeros([n, self.p])
            for i in range(n):
                # construct A_{-i}
                # Xi_nxp has n-1 rows, since Xaug_n1xp didn't include test point in this condition
                Xi_nxp = np.vstack([Xaug_n1xp[: i], Xaug_n1xp[i + 1 :]]) 
                Ai = get_invcov_dot_xt(Xi_nxp, self.gamma, use_lapack=self.use_lapack)
#                 print("Ai dim", np.shape(Ai))
#                 print("Ai[:, : -1] dim", np.shape(Ai[:, : -1]))
#                 print("yi_ dim", np.shape(yi_))
                
#                 a_n1 = C.dot(Xaug_n1xp.T)
#                 b_n1 = A[:, -1].dot(Xaug_n1xp.T)

                # compute linear parameterizations of \mu_{-i, y}(X_i)
                yi_ = np.hstack([ytrain_n[: i], ytrain_n[i + 1 :]])  # n - 1 elements
#                 print("Xaug_n1xp dim", np.shape(Xaug_n1xp))
#                 print("Xi_nxp dim", np.shape(Xi_nxp))
#                 print("Ai dim", np.shape(Ai))
#                 print("Ai[:, : -1] dim", np.shape(Ai[:, : -1]))
#                 print("yi_ dim", np.shape(yi_))
        
                Ci = Ai.dot(yi_) # p elements Ai[:, : -1] changed to Ai
                ai = Ci.dot(Xaug_n1xp[i])  # = Xtrain_nxp[i]
#                 bi = Ai[:, -1].dot(Xaug_n1xp[i])

                # store
                ab_nx1[i] = ai #, bi
                C_nxp[i] = Ci
#                 An_nxp[i] = Ai[:, ]

            # LOO score for i = n + 1
#             tmp = get_invcov_dot_xt(Xaug_n1xp[: -1], self.gamma, use_lapack=self.use_lapack)
#             beta_p = tmp.dot(ytrain_n)
#             alast = beta_p.dot(Xaug_n1xp[-1])  # prediction a_{n + 1}. Xaug_n1xp[-1] = Xtest_p

            # process LOO scores for each candidate value y
#             scoresloo_n1xy = np.zeros([n + 1, self.n_y])
#             by_nxy = np.outer(ab_nx2[:, 1], self.ys)
            prediy_nxy = ab_nx1[:, 0][:, None] # + by_nxy
#             scoresloo_n1xy[: -1] = np.abs(ytrain_n[:, None] - prediy_nxy)
#             scoresloo_n1xy[-1] = np.abs(self.ys - alast)

            ## Compute one-step weights w_n1xy
            scoresloo_n1xy = None
            w_nxy = None
            if compute_lrs:
                
#                 predall_u = self.Xuniv_uxp.dot(beta_p)
#                 Z = np.sum(np.exp(lmbda * predall_u))

                ## Removing the part that is a function of candidate labels
                betaiy_nxpxy = C_nxp[:, :, None] ##+ self.ys * An_nxp[:, :, None] 
                # compute normalizing constant in Eq. 6 in main paper
                pred_nxyxu = np.tensordot(betaiy_nxpxy, self.Xuniv_uxp, axes=(1, 1))
                normconst_nxy = np.sum(np.exp(lmbda * pred_nxyxu), axis=2)
                ptrain_n = self.ptrain_fn(Xaug_n1xp)

                ## Each row of these are the n+1 "one-step" weights for a specific candidate label
                w_nxy = np.zeros([n, self.n_y]) ## replaced n+1 with n
                wi_num_nxy = np.exp(lmbda * prediy_nxy)
                
#                 print("w_nxy dim:", np.shape(w_nxy))
#                 print("wi_num_nxy dim:", np.shape(wi_num_nxy))
#                 print("ptrain_n[:, None] dim:", np.shape(ptrain_n[:, None]))
                
                w_nxy = wi_num_nxy / (ptrain_n[:, None] * normconst_nxy) ## w_n1xy[: -1] replaced with w_n1xy

#                 # for last i = n + 1, which is constant across candidate values of y
#                 Z = self.get_normalizing_constant(beta_p, lmbda)
#                 w_n1xy[-1] = np.exp(lmbda * alast) / (self.ptrain_fn(Xaug_n1xp[-1][None, :]) * Z)

            return scoresloo_n1xy, w_nxy
            
            



    def get_confidence_set(self, Xtrain_nxp, ytrain_n, Xtest_1xp, lmbda, alpha: float = 0.1, n_step_curr=1, use_is_scores: bool = False):
        if (self.p != Xtrain_nxp.shape[1]):
            raise ValueError('Feature dimension {} differs from provided Xuniv_uxp {}'.format(
                Xtrain_nxp.shape[1], self.Xuniv_uxp.shape))
        Xaug_n1xp = np.vstack([Xtrain_nxp, Xtest_1xp])

        # ===== compute scores and weights =====

        # compute in-sample scores
        scoresis_n1xy = self.get_insample_scores(Xaug_n1xp, ytrain_n) if use_is_scores else None

        # compute LOO scores and likelihood ratios
        scoresloo_n1xy, w_n1xy = self.get_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda, n_step_curr=n_step_curr)
        
#         print("scoresloo_n1xy shape : ", np.shape(scoresloo_n1xy))
#         print("w_n1xy shape : ", np.shape(w_n1xy))

        # ===== construct confidence sets =====

        # based on LOO score
        looq_y = get_weighted_quantile(1 - alpha, w_n1xy, scoresloo_n1xy)
        loo_cs = self.ys[scoresloo_n1xy[-1] <= looq_y]

        # based on in-sample score
        is_cs = None
        if use_is_scores:
            isq_y = get_weighted_quantile(1 - alpha, w_n1xy, scoresis_n1xy)
            is_cs = self.ys[scoresis_n1xy[-1] <= isq_y]
        return loo_cs, is_cs, w_n1xy ## Drew added w_n1xy on 20240223


### Drew modified
class JAWRidge(ABC):
    """
    Abstract base class for JAW with ridge regression mu function, based on class for full conformal
    """
    def __init__(self, ptrain_fn, Xuniv_uxp, gamma, use_lapack: bool = True):
        """
        :param ptrain_fn: function that outputs likelihood of input under training input distribution, p_X
        :param ys: numpy array of candidate labels
        :param Xuniv_uxp: (u, p) numpy array encoding all sequences in domain (e.g., all 2^13 sequences
            in Poelwijk et al. 2019 data set), needed for computing normalizing constant
        :param gamma: float, ridge regularization strength
        :param use_lapack: bool, whether or not to use low-level LAPACK functions for inverting covariance (fastest)
        """
        self.ptrain_fn = ptrain_fn
        self.Xuniv_uxp = Xuniv_uxp
        self.p = Xuniv_uxp.shape[1]
#         self.ys = ys ## Drew: maybe don't need this
#         self.n_y = ys.size
        self.gamma = gamma
        self.use_lapack = use_lapack

    def get_normalizing_constant(self, beta_p, lmbda):
        predall_u = self.Xuniv_uxp.dot(beta_p)
        Z = np.sum(np.exp(lmbda * predall_u))
        return Z

    def compute_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda, compute_lrs: bool = True):
        """
        Compute jackknife+ LOO scores, i.e. residuals using model trained on *n-1* data points (n-1 training points, no candidate test points).

        :param Xaug_n1xp: (n + 1, p) numpy array encoding all n + 1 sequences (training + candidate test point)
        :param ytrain_n: (n,) numpy array of true labels for the n training points
        :param lmbda: float, inverse temperature of design algorithm in Eq. 6, {0, 2, 4, 6} in main paper
        :param compute_lrs: bool: whether or not to compute likelihood ratios (this part takes the longest,
            so set to False if only want to compute scores)
        :return: (n + 1, |Y|) numpy arrays of scores S_i(X_test, y) and weights w_i^y(X_test) in Eq. 3 in main paper
        """
        # Compute jackknife+ LOO residuals, test point predictions, and weights
        n = ytrain_n.size
        resids_LOO = np.zeros(n)
        muh_LOO_vals_testpoint = np.zeros(n) ## Only have one testpoint
        
        # Weights
        unnormalized_weights = np.zeros(n + 1)
        for i in range(n):
            ## Create LOO X and y data
            Xi_LOO_n_minus_1xp = np.vstack([Xaug_n1xp[: i], Xaug_n1xp[i + 1 : n]]) ## LOO training data inputs
            yi_LOO_train_n = np.concatenate((ytrain_n[: i], ytrain_n[i + 1 : n])) ## LOO training data outputs
            
            ## Get LOO residuals and test point predictions
            tmp = get_invcov_dot_xt(Xi_LOO_n_minus_1xp, self.gamma, use_lapack=self.use_lapack)
            beta_p = tmp.dot(yi_LOO_train_n)
            muh_i_LOO = beta_p.dot(Xaug_n1xp[i]) ## ith LOO prediction on point i : mu_{-i}(X_i)
            resids_LOO[i] = np.abs(ytrain_n[i] - muh_i_LOO) ## ith LOO residual
            muh_LOO_vals_testpoint[i] = beta_p.dot(Xaug_n1xp[-1]) ## ith LOO prediction on test point n+1 : mu_{-i}(X_{n+1})
            
            ## Calculate unnormalized weights for the training scores 1:n
            unnormalized_weights[i] = (np.exp(lmbda * muh_i_LOO) / self.ptrain_fn(Xaug_n1xp[i][None, :])) * (np.exp(lmbda * muh_LOO_vals_testpoint[i]) / self.ptrain_fn(Xaug_n1xp[-1][None, :]))
                        
        ## Compute jackknife+ upper and lower predictive values
        unweighted_lower_vals = np.zeros(n+1)
        unweighted_upper_vals = np.zeros(n+1)
        unweighted_lower_vals[:n] = muh_LOO_vals_testpoint - resids_LOO
        unweighted_upper_vals[:n] = muh_LOO_vals_testpoint + resids_LOO
        
        
        ## Add infinity
        unweighted_lower_vals[n] = -math.inf
        unweighted_upper_vals[n] = math.inf
        
        
        ## Calculate test point unnormalized weight
        tmp = get_invcov_dot_xt(Xaug_n1xp[: -1], self.gamma, use_lapack=self.use_lapack)
        beta_p = tmp.dot(ytrain_n)
        muh_test = beta_p.dot(Xaug_n1xp[-1])
        unnormalized_weights[n] = (np.exp(lmbda * muh_test) / self.ptrain_fn(Xaug_n1xp[-1][None, :]))**2
        
        weights_normalized = unnormalized_weights / np.sum(unnormalized_weights)
        

        return unweighted_lower_vals, unweighted_upper_vals, weights_normalized

    @abstractmethod
    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        pass

    def get_confidence_set(self, Xtrain_nxp, ytrain_n, Xtest_1xp, lmbda, alpha: float = 0.1, use_is_scores: bool = False):
        if (self.p != Xtrain_nxp.shape[1]):
            raise ValueError('Feature dimension {} differs from provided Xuniv_uxp {}'.format(
                Xtrain_nxp.shape[1], self.Xuniv_uxp.shape))
        Xaug_n1xp = np.vstack([Xtrain_nxp, Xtest_1xp])

        # ===== compute scores and weights =====

        # compute in-sample scores
        scoresis_n1xy = self.get_insample_scores(Xaug_n1xp, ytrain_n) if use_is_scores else None

        # compute LOO scores and likelihood ratios
        unweighted_lower_vals, unweighted_upper_vals, weights_normalized = self.get_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda)

        # ===== construct confidence sets =====
        y_lower = weighted_quantile(unweighted_lower_vals, weights_normalized, alpha)
#         print(weights_normalized)
#         print(y_lower)
        y_upper = weighted_quantile(unweighted_upper_vals, weights_normalized, 1 - alpha)
        
        return y_lower, y_upper


    
    
class ConformalRidgeExchangeable(ConformalRidge):
    """
    Class for full conformal with ridge regression, assuming exchangeable data.
    """
    def __init__(self, ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack: bool = True):
        super().__init__(ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack=use_lapack)

    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda, n_step_curr=0):
        scoresloo_n1xy, _ = self.compute_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda, compute_lrs=False)
        # for exchangeble data, equal weights on all data points (no need to compute likelihood ratios in line above)
        w_n1xy = np.ones([Xaug_n1xp.shape[0], self.n_y])
        return scoresloo_n1xy, w_n1xy


### Drew modified
class ConformalRidgeMultistepFeedbackCovariateShift(ConformalRidge):
    """
    Class for full conformal with ridge regression under multistep feedback covariate shift
    """
    def __init__(self, ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack: bool = True):
        super().__init__(ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack=use_lapack)

    ### Drew modified
    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda, n_step_curr=1):
        scoresloo_n1xy, w_n1xy = self.compute_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda, compute_lrs=True, n_step_curr=n_step_curr)
        return scoresloo_n1xy, w_n1xy
    
    
### Drew modified
class JAWRidgeFeedbackCovariateShift(JAWRidge):
    """
    Class for JAW with ridge regression under feedback covariate shift
    """
    def __init__(self, ptrain_fn, Xuniv_uxp, gamma, use_lapack: bool = True):
        super().__init__(ptrain_fn, Xuniv_uxp, gamma, use_lapack=use_lapack)

    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        unweighted_lower_vals, unweighted_upper_vals, weights_normalized = self.compute_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda, compute_lrs=True)
        return unweighted_lower_vals, unweighted_upper_vals, weights_normalized


class ConformalRidgeStandardCovariateShift(ConformalRidge):
    """
    Class for full conformal with ridge regression under standard covariate shift.
    """
    def __init__(self, ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack: bool = True):
        super().__init__(ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack=use_lapack)

    def get_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        # fit model to training data
        tmp = get_invcov_dot_xt(Xaug_n1xp[: -1], self.gamma, use_lapack=self.use_lapack)
        beta_p = tmp.dot(ytrain_n)

        # compute normalizing constant for test covariate distribution
        Z = self.get_normalizing_constant(beta_p, lmbda)

        # get likelihood ratios for n + 1 covariates
        pred_n1 = Xaug_n1xp.dot(beta_p)
        ptest_n1 = np.exp(lmbda * pred_n1) / Z
        w_n1 = ptest_n1 / self.ptrain_fn(Xaug_n1xp)
        return w_n1

    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda, n_step_curr=1):
        # LOO scores
        scoresloo_n1xy, _ = self.compute_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda, compute_lrs=False)

        # compute likelihood ratios
        w_n1 = self.get_lrs(Xaug_n1xp, ytrain_n, lmbda)
        w_n1xy = w_n1[:, None] * np.ones([Xaug_n1xp.shape[0], self.n_y])
        return scoresloo_n1xy, w_n1xy



# ========== utilities and classes for full conformal with black-box model ==========

def get_scores(model, Xaug_nxp, yaug_n, use_loo_score: bool = False):
    if use_loo_score:
        n1 = yaug_n.size  # n + 1
        scores_n1 = np.zeros([n1])

        for i in range(n1):
            Xtrain_nxp = np.vstack([Xaug_nxp[: i], Xaug_nxp[i + 1 :]])
            ytrain_n = np.hstack([yaug_n[: i], yaug_n[i + 1 :]])

            # train on LOO dataset
            model.fit(Xtrain_nxp, ytrain_n)
            pred_1 = model.predict(Xaug_nxp[i][None, :])
            scores_n1[i] = np.abs(yaug_n[i] - pred_1[0])

    else:  # in-sample score
        model.fit(Xaug_nxp, yaug_n)
        pred_n1 = model.predict(Xaug_nxp)
        scores_n1 = np.abs(yaug_n - pred_n1)
    return scores_n1


class Conformal(ABC):
    """
    Abstract base class for full conformal with black-box predictive model.
    """
    def __init__(self, model, ptrain_fn, ys, Xuniv_uxp):
        """
        :param model: object with predict() method
        :param ptrain_fn: function that outputs likelihood of input under training input distribution, p_X
        :param ys: (|Y|,) numpy array of candidate labels
        :param Xuniv_uxp: (u, p) numpy array encoding all sequences in domain (e.g., all 2^13 sequences
            in Poelwijk et al. 2019 data set), needed for computing normalizing constant
        """
        self.model = model
        self.ptrain_fn = ptrain_fn
        self.ys = ys
        self.Xuniv_uxp = Xuniv_uxp
        self.p = Xuniv_uxp.shape[1]
        self.n_y = ys.size

    @abstractmethod
    def get_lrs(self, Xaug_n1xp, yaug_n1, lmbda):
        pass

    def get_confidence_set(self, Xtrain_nxp, ytrain_n, Xtest_1xp, lmbda,
                           use_loo_score: bool = True, alpha: float = 0.1, print_every: int = 10, verbose: bool = True):
        if (self.p != Xtrain_nxp.shape[1]):
            raise ValueError('Feature dimension {} differs from provided Xuniv_uxp {}'.format(
                Xtrain_nxp.shape[1], self.Xuniv_uxp.shape))

        np.set_printoptions(precision=3)
        cs, n = [], ytrain_n.size
        t0 = time.time()
        Xaug_n1xp = np.vstack([Xtrain_nxp, Xtest_1xp])
        scores_n1xy = np.zeros([n + 1, self.n_y])
        w_n1xy = np.zeros([n + 1, self.n_y])

        for y_idx, y in enumerate(self.ys):

            # get scores
            yaug_n1 = np.hstack([ytrain_n, y])
            scores_n1 = get_scores(self.model, Xaug_n1xp, yaug_n1, use_loo_score=use_loo_score)
            scores_n1xy[:, y_idx] = scores_n1

            # get likelihood ratios
            w_n1 = self.get_lrs(Xaug_n1xp, yaug_n1, lmbda)
            w_n1xy[:, y_idx] = w_n1

            # for each value of inverse temperature lambda, compute quantile of weighted scores
            q = get_weighted_quantile(1 - alpha, w_n1, scores_n1)

            # if y <= quantile, include in confidence set
            if scores_n1[-1] <= q:
                cs.append(y)

            # print progress
            if verbose and (y_idx + 1) % print_every == 0:
                print("Done with {} / {} y values ({:.1f} s)".format(
                    y_idx + 1, self.ys.size, time.time() - t0))
        return np.array(cs), scores_n1xy, w_n1xy

    
    
    
### 
### 
### 
### ACTIVE LEARNING EXPERIMENTS
### 
### 
### 
class JAW_FCS_ACTIVE(ABC):
    """
    Abstract base class for JAW with ridge regression mu function, based on class for full conformal
    """
    def __init__(self, model, ptrain_fn, Xuniv_uxp):
        """
        :param model: object with predict() method
        :param ptrain_fn: function that outputs likelihood of input under training input distribution, p_X
        :param Xuniv_uxp: (u, p) numpy array encoding all sequences in domain (e.g., all 2^13 sequences
            in Poelwijk et al. 2019 data set), needed for computing normalizing constant
        """
        self.model = model
        self.ptrain_fn = ptrain_fn
        self.Xuniv_uxp = Xuniv_uxp
        self.p = Xuniv_uxp.shape[1]

    def get_normalizing_constant(self, beta_p, lmbda):
        predall_u = self.Xuniv_uxp.dot(beta_p)
        Z = np.sum(np.exp(lmbda * predall_u))
        return Z

    def compute_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        """
        Compute jackknife+ LOO scores, i.e. residuals using model trained on *n-1* data points (n-1 training points, no candidate test points).

        :param Xaug_n1xp: (n + 1, p) numpy array encoding all n + 1 sequences (training + candidate test point)
        :param ytrain_n: (n,) numpy array of true labels for the n training points
        :param lmbda: float, inverse temperature of design algorithm in Eq. 6, {0, 2, 4, 6} in main paper
        :param compute_lrs: bool: whether or not to compute likelihood ratios (this part takes the longest,
            so set to False if only want to compute scores)
        :return: (n + 1, |Y|) numpy arrays of scores S_i(X_test, y) and weights w_i^y(X_test) in Eq. 3 in main paper
        """
        # Compute jackknife+ LOO residuals, test point predictions, and weights
        n = ytrain_n.size
        n1 = len(Xaug_n1xp) - n
        resids_LOO = np.zeros(n)
        muh_LOO_vals_testpoint = np.zeros((n,n1))
        
        # Oracle weights
        unnormalized_weights_JAW_FCS = np.zeros((n + 1, n1))
        unnormalized_weights_JAW_SCS = np.zeros((n + 1, n1))
        
        
        for i in range(n):
            ## Create LOO X and y data
            Xi_LOO_n_minus_1xp = np.vstack([Xaug_n1xp[: i], Xaug_n1xp[i + 1 : n]]) ## LOO training data inputs
            yi_LOO_train_n = np.concatenate((ytrain_n[: i], ytrain_n[i + 1 : n])) ## LOO training data outputs
            
            ## Get LOO residuals and test point predictions
            self.model.fit(Xi_LOO_n_minus_1xp, yi_LOO_train_n)
            muh_i_LOO = self.model.predict(Xaug_n1xp[i].reshape(1, -1)) ## ith LOO prediction on point i : mu_{-i}(X_i)
            resids_LOO[i] = np.abs(ytrain_n[i] - muh_i_LOO) ## ith LOO residual
            muh_LOO_vals_testpoint[i] = self.model.predict(Xaug_n1xp[-n1:]).T ## ith LOO prediction on test point n+1 : mu_{-i}(X_{n+1})
            
            ## Estimated weights (logistic regression and random forest)
            source_target_labels = np.concatenate([np.zeros(len(Xi_LOO_n_minus_1xp)), np.ones(len(X1))])
            weights_lr = logistic_regression_weight_est(X_full, source_target_labels)
            weights_rf = random_forest_weight_est(X_full, source_target_labels)
        
        
            ## Calculate unnormalized likelihoo-ratio weights for FCS
            unnormalized_weights_JAW_FCS[i] = (np.exp(lmbda * muh_i_LOO) / (self.ptrain_fn(Xaug_n1xp[i][None, :]))) * (np.exp(lmbda * muh_LOO_vals_testpoint[i]) / (self.ptrain_fn(Xaug_n1xp[-n1:][None, :])))
            
            
        for j in range(n1):
            ## Calculate unnormalized likelihoo-ratio weights for SCS
            unnormalized_weights_JAW_SCS[:, j] = self.get_lrs(Xaug_n1xp, ytrain_n, lmbda)
            
            
                        
        ## Compute jackknife+ upper and lower predictive values
        unweighted_lower_vals = (muh_LOO_vals_testpoint.T - resids_LOO).T
        unweighted_upper_vals = (muh_LOO_vals_testpoint.T + resids_LOO).T
        
        
        ## Add infinity
        unweighted_lower_vals = np.vstack((unweighted_lower_vals, -math.inf*np.ones(n1)))
        unweighted_upper_vals = np.vstack((unweighted_upper_vals, math.inf*np.ones(n1)))
        
        
        ## Calculate test point unnormalized weight
        self.model.fit(Xaug_n1xp[: -n1], ytrain_n)
        muh_test = self.model.predict(Xaug_n1xp[-n1:])
        unnormalized_weights_JAW_FCS[n] = (np.exp(lmbda * muh_test) / self.ptrain_fn(Xaug_n1xp[-n1:][None, :]))**2
        
        weights_normalized_JAW_FCS = np.zeros((n + 1, n1))
        weights_normalized_JAW_SCS = np.zeros((n + 1, n1))
        for j in range(0, n1):
            weights_normalized_JAW_FCS[:,j] = unnormalized_weights_JAW_FCS[:,j] / np.sum(unnormalized_weights_JAW_FCS[:,j])
            weights_normalized_JAW_SCS[:,j] = unnormalized_weights_JAW_SCS[:,j] / np.sum(unnormalized_weights_JAW_SCS[:,j])
        
        return unweighted_lower_vals, unweighted_upper_vals, weights_normalized_JAW_FCS, weights_normalized_JAW_SCS

    @abstractmethod
    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        pass

    def get_confidence_set(self, Xtrain_nxp, ytrain_n, Xtest_1xp, lmbda, alpha: float = 0.1):
        if (self.p != Xtrain_nxp.shape[1]):
            raise ValueError('Feature dimension {} differs from provided Xuniv_uxp {}'.format(
                Xtrain_nxp.shape[1], self.Xuniv_uxp.shape))
        Xaug_n1xp = np.vstack([Xtrain_nxp, Xtest_1xp])
        n1 = len(Xtest_1xp)

        # ===== compute scores and weights =====

        # compute LOO scores and likelihood ratios
        unweighted_lower_vals, unweighted_upper_vals, weights_normalized_JAW_FCS, weights_normalized_JAW_SCS = self.get_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda)

        # ===== construct confidence intervals for FCS and SCS =====
        y_lower_JAW_FCS = np.zeros(n1)
        y_upper_JAW_FCS = np.zeros(n1)
        y_lower_JAW_SCS = np.zeros(n1)
        y_upper_JAW_SCS = np.zeros(n1)
        y_lower_Jplus = np.zeros(n1)
        y_upper_Jplus = np.zeros(n1)
        uniform_weights = np.ones(n+1) / (n+1)
        for j in range(0, n1):
            y_lower_JAW_FCS[j] = weighted_quantile(unweighted_lower_vals[:, j], weights_normalized_JAW_FCS[:, j], alpha)
            y_upper_JAW_FCS[j] = weighted_quantile(unweighted_upper_vals[:, j], weights_normalized_JAW_FCS[:, j], 1 - alpha)
            y_lower_JAW_SCS[j] = weighted_quantile(unweighted_lower_vals[:, j], weights_normalized_JAW_SCS[:, j], alpha)
            y_upper_JAW_SCS[j] = weighted_quantile(unweighted_upper_vals[:, j], weights_normalized_JAW_SCS[:, j], 1 - alpha)
            y_lower_Jplus[j] = weighted_quantile(unweighted_lower_vals[:, j], uniform_weights, alpha)
            y_upper_Jplus[j] = weighted_quantile(unweighted_upper_vals[:, j], uniform_weights, 1 - alpha)
            
        return y_lower_JAW_FCS, y_upper_JAW_FCS, y_lower_JAW_SCS, y_upper_JAW_SCS, y_lower_Jplus, y_upper_Jplus

    
        
    
    def compute_PIs_active(self, Xtrain_nxp, ytrain_n, Xtest_1xp, ytest_n1, Xtrain_split, Xcal_split, ytrain_split, ycal_split, Xtest_n1xp_split, ytest_n1_split, Xpool_split, w_split_mus_prev_steps, exp_vals_pool_list_of_vecs_all_steps, method_names, n_step_curr, X_dataset, n_cal_initial, alpha_aci_curr, weight_bounds, weight_adj_depth_maxes = [1,2], tilt_factor = 1/10, bandwidth = 1.0, beta: float = 1.0, alpha: float = 0.1, K_vals = [8, 12, 16, 24, 32, 48], n_train_initial=100, n_dataset = None, replacement=True):
        # , add_to_cal=True
        if (self.p != Xtrain_nxp.shape[1]):
            raise ValueError('Feature dimension {} differs from provided Xuniv_uxp {}'.format(
                Xtrain_nxp.shape[1], self.Xuniv_uxp.shape))
#         Xaug_n1xp = np.vstack([Xtrain_nxp, Xtest_1xp])
        Xaug_cal_test_split = np.vstack([Xcal_split, Xtest_n1xp_split])
        n = ytrain_n.size
#         n1 = len(Xaug_n1xp) - n ### Temp removed this 20240111, as chnaged split test points to those queried
        n1 = len(Xtest_n1xp_split)
#         n1 = len(ytest_n1_split)

        
        ###############################
        # split conformal
        ###############################
        n_cal = len(ycal_split)
        muh_split = self.model.fit(Xtrain_split, ytrain_split)
#         muh_split = w_split_mus_prev_steps[-1] ## Checking what happens when avoid refitting
        muh_split_vals = muh_split.predict(np.r_[Xcal_split,Xtest_n1xp_split]) # , std_split_vals
        resids_split = np.abs(ycal_split-muh_split_vals[:n_cal])
        muh_split_vals_testpoint = muh_split_vals[n_cal:]
        ind_split = (np.ceil((1-alpha)*(n_cal+1))).astype(int)
        
        # print("split interval computed : ", muh_split_vals_testpoint - np.sort(resids_split)[ind_split-1], \
        #                muh_split_vals_testpoint + np.sort(resids_split)[ind_split-1])
        
        resids_split_aci = np.concatenate([resids_split,])
        
        PIs_dict = {'split' : pd.DataFrame(\
                np.c_[muh_split_vals_testpoint - np.sort(resids_split)[ind_split-1], \
                       muh_split_vals_testpoint + np.sort(resids_split)[ind_split-1]],\
                        columns = ['lower','upper'])}
        
    
        


        
        w_split_mus_prev_steps.append(deepcopy(self.model))
        w_split_mus_prev_and_curr_steps = deepcopy(w_split_mus_prev_steps)
#         w_split_mus_prev_and_curr_steps.append(deepcopy(self.model))

        
        if (n_cal + 1 != len(muh_split_vals)):
            raise ValueError('n_cal + 1 != len(muh_split_vals); code not yet set up for batch setting')
            

        
        ## Using previous muh functions:
        ## Compute matrix containing the weights (for cal and test pts; as well as for pool) at each step
        exp_vals_cal_test_MAT_all_steps = np.zeros((n_step_curr, n_cal + 1))
        # exp_vals_pool_sum_all_steps = np.zeros((n_step_curr, len(Xpool_split)))
        exp_vals_pool_sum_all_steps = np.zeros(n_step_curr)

        
        for i, muh_split_curr in enumerate(w_split_mus_prev_and_curr_steps):
            
            B = weight_bounds[i]
            
            ## Compute (unnormalized) total pool weights
            _, std_pool_muh_curr_ = muh_split_curr.predict(Xpool_split, return_std=True)
            std_pool_muh_curr = get_f_std(std_pool_muh_curr_, muh_split_curr)
            
            var_pool_muh_curr = std_pool_muh_curr**2
            var_pool_muh_curr_minmax_normed = (var_pool_muh_curr) / (max(var_pool_muh_curr) - min(var_pool_muh_curr))
            exp_vars_pool = np.exp(var_pool_muh_curr_minmax_normed * tilt_factor)
            # exp_vals_pool_MAT_all_steps[i] = np.sort(exp_vars_pool)


            
            ## Compute (unnormalized) total calibration weights
            _, std_muh_curr_ = muh_split_curr.predict(np.r_[Xcal_split,Xtest_n1xp_split], return_std=True)
            std_muh_curr = get_f_std(std_muh_curr_, muh_split_curr)
            
            var_cal_test_muh_curr = std_muh_curr**2
            var_cal_test_muh_curr_minmax_normed = (var_cal_test_muh_curr)  / (max(var_pool_muh_curr) - min(var_pool_muh_curr))
            # exp_vals_cal_test_MAT_all_steps[i] = np.exp(var_cal_test_muh_curr_minmax_normed * tilt_factor)
            
            
            
            exp_vals_cal_test_MAT_all_steps[i] = np.minimum(np.exp(var_cal_test_muh_curr_minmax_normed * tilt_factor), B)
            
            
        
            
            if (np.sum(exp_vars_pool) != np.sum(exp_vals_pool_list_of_vecs_all_steps[i])):
                print("Warning! np.sum(exp_vars_pool) = ", np.sum(exp_vars_pool), "!= np.sum(exp_vals_pool_list_of_vecs_all_steps[i])", np.sum(exp_vals_pool_list_of_vecs_all_steps[i]))
                
            
            
            exp_vals_pool_sum_all_steps[i] = np.sum(np.minimum(exp_vals_pool_list_of_vecs_all_steps[i], B)) # np.sum(np.minimum(exp_vars_pool, B))
                    
                
            

            
        ### **** Adjusted weight computations ****
        wsplit_quantiles_lower_list = []
        wsplit_quantiles_upper_list = []
        
#         print("weight_adj_depth_maxes : ", weight_adj_depth_maxes)

        weights_normalized_wsplit_all = []
        
        for depth_max in weight_adj_depth_maxes:
            
            depth_max_curr = min(depth_max, n_step_curr)
            
            time_begin_w = time.time()
            
            if (replacement):
                ## If sampling with replacement
                # Z = pool_weights_totals_prev_steps[-1]
                
                
                ## Note: For replacement case, easier to normalize ahead of time by dividing by Z
                SCS_split_weights_vec = compute_w_ptest_split_active_replacement(exp_vals_cal_test_MAT_all_steps, n_step_curr=depth_max_curr)
                


            else:
#                 Z = pool_weights_totals_prev_steps[-1]
#                 ## Note: For replacement case, easier to normalize ahead of time by dividing by Z
#                 SCS_split_weights_vec = compute_w_ptest_split_active_replacement(cal_test_vals_mat = w_split_MAT_all_steps/Z, n_step_curr=depth_max_curr)
                ## Else sampling without replacement
                SCS_split_weights_vec = compute_w_ptest_split_active_no_replacement(cal_test_vals_mat = w_split_MAT_all_steps, n_step_curr = depth_max_curr, pool_weight_arr_curr = pool_weights_totals_prev_steps, n_pool_curr = len(Xpool_split)) #

                
                
            print("Time elapsed for depth ", depth_max_curr, " (min) : ", (time.time() - time_begin_w) / 60)
            
            
            SCS_split_weights_vec = SCS_split_weights_vec.flatten()

            
       
            ## Add infty (distribution on augmented real line)
            positive_infinity = np.array([float('inf')])
            unweighted_split_vals = np.concatenate([resids_split, positive_infinity])
            
            
            ## Store weight values for plotting figures later
            # print("depth_max : ", depth_max)
            # print("unweighted_split_vals shape: ", np.shape(unweighted_split_vals))
            # print("SCS_split_weights_vec shape: ", np.shape(SCS_split_weights_vec))
            # print("SCS_split_weights_vec :", SCS_split_weights_vec)
            
            

            wsplit_quantiles = np.zeros(n1)

            weights_normalized_wsplit = np.zeros((n_cal + 1, n1))
            sum_cal_weights = np.sum(SCS_split_weights_vec[:n_cal])

            for j in range(0, n1):
                for i in range(0, n_cal + 1):
                    if (i < n_cal):
                        weights_normalized_wsplit[i, j] = SCS_split_weights_vec[i] / (sum_cal_weights + SCS_split_weights_vec[n_cal + j])
                    else:
                        weights_normalized_wsplit[i, j] = SCS_split_weights_vec[n_cal+j] / (sum_cal_weights + SCS_split_weights_vec[n_cal + j])



            weights_normalized_wsplit_all.append(np.concatenate([sort_both_by_first(unweighted_split_vals[0:n_cal_initial], weights_normalized_wsplit[0:n_cal_initial,0])[1], weights_normalized_wsplit[n_cal_initial:,0]]))
    
    
            wsplit_quantiles_lower = np.zeros(n1)
            wsplit_quantiles_upper = np.zeros(n1)
            for j in range(0, n1):
                wsplit_quantiles[j] = weighted_quantile(unweighted_split_vals, weights_normalized_wsplit[:, j], 1 - alpha)

            
            wsplit_quantiles_lower_list.append(wsplit_quantiles_lower)
            wsplit_quantiles_upper_list.append(wsplit_quantiles_upper)
            
        
        
            PIs_dict['wsplit_' + str(depth_max)] = pd.DataFrame(np.c_[muh_split_vals_testpoint - wsplit_quantiles, \
                                               muh_split_vals_testpoint + wsplit_quantiles],\
                                               columns = ['lower','upper'])


        
        
        ###### ACI ######
        # print("1-alpha_aci_curr : ", 1-alpha_aci_curr)
        
        q_aci = np.quantile(unweighted_split_vals, 1-alpha_aci_curr)
        
        
        
        
        PIs_dict['aci'] = pd.DataFrame(np.c_[muh_split_vals_testpoint - q_aci, \
                           muh_split_vals_testpoint + q_aci],\
                            columns = ['lower','upper'])
        
        
        
        
        return PIs_dict, w_split_mus_prev_steps, weights_normalized_wsplit_all
    
    

class ConformalExchangeable(Conformal):
    """
    Full conformal with black-box predictive model, assuming exchangeable data.
    """
    def __init__(self, model, ptrain_fn, ys, Xuniv_uxp):
        super().__init__(model, ptrain_fn, ys, Xuniv_uxp)

    def get_lrs(self, Xaug_n1xp, yaug_n1, lmbda):
        return np.ones([Xaug_n1xp.shape[0]])


class ConformalFeedbackCovariateShift(Conformal):
    """
    Full conformal with black-box predictive model under feedback covariate shift via Eq. 6 in main paper.
    """
    def __init__(self, model, ptrain_fn, ys, Xuniv_uxp):
        super().__init__(model, ptrain_fn, ys, Xuniv_uxp)

    def get_lrs(self, Xaug_n1xp, yaug_n1, lmbda):
        # compute weights for each value of lambda, the inverse temperature
        w_n1 = np.zeros([yaug_n1.size])
        for i in range(yaug_n1.size):

            # fit LOO model
            Xtr_nxp = np.vstack([Xaug_n1xp[: i], Xaug_n1xp[i + 1 :]])
            ytr_n = np.hstack([yaug_n1[: i], yaug_n1[i + 1 :]])
            self.model.fit(Xtr_nxp, ytr_n)

            # compute normalizing constant
            predall_n = self.model.predict(self.Xuniv_uxp)
            Z = np.sum(np.exp(lmbda * predall_n))

            # compute likelihood ratios
            testpred = self.model.predict(Xaug_n1xp[i][None, :])
            ptest = np.exp(lmbda * testpred) / Z
            w_n1[i] = ptest / self.ptrain_fn(Xaug_n1xp[i][None, :])
        return w_n1


### Drew modified
class JAWFeedbackCovariateShift(JAW_FCS):
    """
    Class for JAW with ridge regression under feedback covariate shift
    """
    def __init__(self, model, ptrain_fn, Xuniv_uxp):
        super().__init__(model, ptrain_fn, Xuniv_uxp)

    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        unweighted_lower_vals, unweighted_upper_vals, weights_normalized = self.compute_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda)
        return unweighted_lower_vals, unweighted_upper_vals, weights_normalized

    def get_lrs(self, Xaug_n1xp, ytrain_n, lmbda, split=False, idx=None): ### THIS IS GOING TO HAVE TO TAKE AS ARGUMENTS THIS SPLITS THEMSELVES
        n = len(ytrain_n)
        n1 = len(Xaug_n1xp) - n
        
        if (split == True):
            n_half = int(np.floor(n/2))
            idx_train, idx_cal = idx[:n_half], idx[n_half:]
            self.model.fit(Xaug_n1xp[idx_train],ytrain_n[idx_train])
            pred_n1 = self.model.predict(Xaug_n1xp)
            ptest_n1 = np.exp(lmbda * pred_n1)
            w_n1 = ptest_n1 / self.ptrain_fn(Xaug_n1xp)
            return w_n1

        else:
            self.model.fit(Xaug_n1xp[: -n1], ytrain_n)  # Xtrain_nxp, ytrain_n
            # get likelihood ratios
            pred_n1 = self.model.predict(Xaug_n1xp)
            ptest_n1 = np.exp(lmbda * pred_n1)
            w_n1 = ptest_n1 / self.ptrain_fn(Xaug_n1xp)
            return w_n1


### Drew modified
class JAWFeedbackCovariateShiftActive(JAW_FCS_ACTIVE):
    """
    Class for JAW with ridge regression under feedback covariate shift
    """
    def __init__(self, model, ptrain_fn, Xuniv_uxp):
        super().__init__(model, ptrain_fn, Xuniv_uxp)

    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        unweighted_lower_vals, unweighted_upper_vals, weights_normalized = self.compute_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda)
        return unweighted_lower_vals, unweighted_upper_vals, weights_normalized

    def get_lrs(self, Xaug_n1xp, ytrain_n, lmbda, split=False, idx=None): ### THIS IS GOING TO HAVE TO TAKE AS ARGUMENTS THIS SPLITS THEMSELVES
        n = len(ytrain_n)
        n1 = len(Xaug_n1xp) - n
        
        if (split == True):
            n_half = int(np.floor(n/2))
            idx_train, idx_cal = idx[:n_half], idx[n_half:]
            self.model.fit(Xaug_n1xp[idx_train],ytrain_n[idx_train])
            pred_n1 = self.model.predict(Xaug_n1xp)
            ptest_n1 = np.exp(lmbda * pred_n1)
            w_n1 = ptest_n1 / self.ptrain_fn(Xaug_n1xp)
            return w_n1

        else:
            self.model.fit(Xaug_n1xp[: -n1], ytrain_n)  # Xtrain_nxp, ytrain_n
            # get likelihood ratios
            pred_n1 = self.model.predict(Xaug_n1xp)
            ptest_n1 = np.exp(lmbda * pred_n1)
            w_n1 = ptest_n1 / self.ptrain_fn(Xaug_n1xp)
            return w_n1

        
### Drew modified
class SplitFeedbackCovariateShift(JAW_FCS):
    """
    Class for JAW with ridge regression under feedback covariate shift
    """
    def __init__(self, model, ptrain_fn, Xuniv_uxp):
        super().__init__(model, ptrain_fn, Xuniv_uxp)

    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        unweighted_lower_vals, unweighted_upper_vals, weights_normalized = self.compute_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda)
        return unweighted_lower_vals, unweighted_upper_vals, weights_normalized

    def get_lrs(self, Xaug_n1xp, ytrain_n, lmbda, split=False, idx=None): ### THIS IS GOING TO HAVE TO TAKE AS ARGUMENTS THIS SPLITS THEMSELVES
        n = len(ytrain_n)
        n1 = len(Xaug_n1xp) - n
        
        if (split == True):
            n_half = int(np.floor(n/2))
            idx_train, idx_cal = idx[:n_half], idx[n_half:]
            self.model.fit(Xaug_n1xp[idx_train],ytrain_n[idx_train])
            pred_n1 = self.model.predict(Xaug_n1xp)
            ptest_n1 = np.exp(lmbda * pred_n1)
            w_n1 = ptest_n1 / self.ptrain_fn(Xaug_n1xp)
            return w_n1

        else:
            self.model.fit(Xaug_n1xp[: -n1], ytrain_n)  # Xtrain_nxp, ytrain_n
            # get likelihood ratios
            pred_n1 = self.model.predict(Xaug_n1xp)
            ptest_n1 = np.exp(lmbda * pred_n1)
            w_n1 = ptest_n1 / self.ptrain_fn(Xaug_n1xp)
            return w_n1


class ConformalFeedbackCovariateShift(Conformal):
    """
    Full conformal with black-box predictive model under feedback covariate shift via Eq. 6 in main paper.
    """
    def __init__(self, model, ptrain_fn, ys, Xuniv_uxp):
        super().__init__(model, ptrain_fn, ys, Xuniv_uxp)

    def get_lrs(self, Xaug_n1xp, yaug_n1, lmbda):
        # compute weights for each value of lambda, the inverse temperature
        w_n1 = np.zeros([yaug_n1.size])
        for i in range(yaug_n1.size):

            # fit LOO model
            Xtr_nxp = np.vstack([Xaug_n1xp[: i], Xaug_n1xp[i + 1 :]])
            ytr_n = np.hstack([yaug_n1[: i], yaug_n1[i + 1 :]])
            self.model.fit(Xtr_nxp, ytr_n)

            # compute normalizing constant
            predall_n = self.model.predict(self.Xuniv_uxp)
            Z = np.sum(np.exp(lmbda * predall_n))

            # compute likelihood ratios
            testpred = self.model.predict(Xaug_n1xp[i][None, :])
            ptest = np.exp(lmbda * testpred) / Z
            w_n1[i] = ptest / self.ptrain_fn(Xaug_n1xp[i][None, :])
        return w_n1


class ConformalStandardCovariateShift(Conformal):
    """
    Full conformal with black-box predictive model under standard covariate shift.
    """
    def __init__(self, model, ptrain_fn, ys, Xuniv_uxp):
        super().__init__(model, ptrain_fn, ys, Xuniv_uxp)

    def get_lrs(self, Xaug_n1xp, yaug_n1, lmbda):
        # get normalization constant for test covariate distribution
        self.model.fit(Xaug_n1xp[: -1], yaug_n1[: -1])  # Xtrain_nxp, ytrain_n
        predall_u = self.model.predict(self.Xuniv_uxp)
        Z = np.sum(np.exp(lmbda * predall_u))

        # get likelihood ratios
        pred_n1 = self.model.predict(Xaug_n1xp)
        ptest_n1 = np.exp(lmbda * pred_n1) / Z
        w_n1 = ptest_n1 / self.ptrain_fn(Xaug_n1xp)
        return w_n1

    
