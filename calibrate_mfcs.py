"""
Classes for full and split conformal prediction for exchangeable and multistep feedback covariate shift data.
Full CP implementations are computationally optimized for ridge regression.
Throughout this file, variable name suffixes denote the shape of the numpy array, where
    n: number of training points, or generic number of data points
    n1: n + 1
    p: number of features
    y: number of candidate labels, |Y|
"""

import numpy as np
import time
import scipy as sc

from abc import ABC, abstractmethod

import math
import pandas as pd
import random
from sklearn.neighbors import KernelDensity
from copy import deepcopy
import datetime



def get_f_std(y_std, gpr):
    params = gpr.kernel_.get_params()

    normalized_noise_var = params["k2__noise_level"]
    y_train_var = gpr._y_train_std ** 2

    y_pred_var = y_std ** 2
    f_pred_var = y_pred_var - (y_train_var * normalized_noise_var)
    f_std = np.sqrt(f_pred_var)
    return f_std



## Utilities for computing the factorized likelihood for MFCS Split CP, i.e., the numerator of Eq. (9); 
## the recursive implementation here corresponds to Eq. (16) in Appendix B.2.

def compute_w_ptest_split_active_replacement(cal_test_vals_mat, depth_max):
    '''
        Computes the estimated MFCS Split CP weights for calibration and test points 
        (i.e., numerator in Eq. (9) in main paper or Eq. (16) in Appendix B.2)
        
        @param : cal_test_vals_mat    : (float) matrix of weights with dim (depth_max, n_cal + 1).
                ## For t \in {1, ..., depth_max} : cal_test_vals_mat[t-1, j-1] = w_{n+t}(X_j) = exp(\lambda * \hat{\sigma^2}(X_j))
                ## where X_j is a calibration point for j \in {1, ..., n_cal} and the test point for j=n_cal + 1
                
        @param : depth_max          : (int) indicating the maximum recursion depth
        
        :return: Unnormalized weights on calibration and test points, computed for recursion depth depth_max
    '''
    if (depth_max < 1):
        raise ValueError('Error: depth_max should be an integer >= 1. Currently, depth_max=' + str(depth_max))
      
    if (depth_max == 1):
        ## 
        return cal_test_vals_mat[-1]
        
    n_cal_test = np.shape(cal_test_vals_mat)[1]
    adjusted_vals = deepcopy(cal_test_vals_mat[-1])
    idx_include = np.repeat(True, n_cal_test)
    
    
    for i in range(n_cal_test):
        idx_include[i] = False
        idx_include[i-1] = True
        summation = compute_w_ptest_split_active_replacement_helper(cal_test_vals_mat[:-1,idx_include], depth_max-1)
        adjusted_vals[i] = adjusted_vals[i] * summation
    return adjusted_vals
            
        
def compute_w_ptest_split_active_replacement_helper(cal_test_vals_mat, depth_max):
    '''
        Helper function for "compute_w_ptest_split_active_replacement". Computes a summation such as the two sums in the numerator in equation (7) in paper
        
        @param : cal_test_vals_mat    : (float) matrix of weights with dim (depth_max, n_cal + 1).
                ## For t \in {1, ..., depth_max} : cal_test_vals_mat[t-1, j-1] = w_{n+t}(X_j) = exp(\lambda * \hat{\sigma^2}(X_j))
                ## where X_j is a calibration point for j \in {1, ..., n_cal} and the test point for j=n_cal + 1
                
        @param : depth_max          : (int) indicating the maximum recursion depth
        
        :return: Summation such as the two sums in the numerator in equation (7) in paper
    '''
    if (depth_max == 1):
        return np.sum(cal_test_vals_mat)
    
    else:
        summation = 0
        n_cal_test = np.shape(cal_test_vals_mat)[1]
        idx_include = np.repeat(True, n_cal_test)
        for i in range(n_cal_test):
            idx_include[i] = False
            idx_include[i-1] = True
            summation += cal_test_vals_mat[-1,i]*compute_w_ptest_split_active_replacement_helper(cal_test_vals_mat[:-1,idx_include], depth_max - 1) 
        return summation
    



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




class FullConformalRidge(ABC):
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

    
    

    def compute_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda, compute_weights: bool = True, depth_max=1):
        """
        Compute LOO scores, i.e. residuals using model trained on n data points (training + candidate test points,
        but leave i-th training point out).

        :param Xaug_n1xp: (n + 1, p) numpy array encoding all n + 1 sequences (training + candidate test point)
        :param ytrain_n: (n,) numpy array of true labels for the n training points
        :param lmbda: float, shift magnitude (inverse temperature of design algorithm)
        :param compute_weights: bool: whether or not to compute MFCS CP weights for calibration and test points (this part takes the longest,
            so set to False if only want to compute scores)
        :return: (n + 1, |Y|) numpy arrays of scores S_i(X_test, y) and weights w_i^y(X_test) in Eq. 3 in main paper
        """
        if (depth_max < 1):
            raise ValueError('Error: depth_max should be an integer >= 1. Currently, depth_max =' + str(depth_max))


        ## Depth = 1 in the recursion (see Eq. (16) in Appendix B.2) begins by computing the "one-step" FCS leave-one-out scores and weights.
        scoresloo_n1xy, w_n1xy = self.compute_onestep_full_cp_loo_scores_weights_ridge(Xaug_n1xp, ytrain_n, lmbda, compute_weights)
        
        
        ## Compute weights (Eq. (16) in Appendix B.2) for each candidate value y
        w_n1xy_adjusted = None ## w_n1xy_adjusted will keep track of the weights adjusted for estimation depth $d$
        if compute_weights:

            if (depth_max == 1):
                ## If estimation depth == 1, then return the scores and one-step weights
                return scoresloo_n1xy, w_n1xy

            ## Otherwise estimation depth >= 2, and need to adjust the weights further
            w_n1xy_adjusted = deepcopy(w_n1xy)
            idx_include_y = np.repeat(True, len(ytrain_n))
            idx_include_x = np.repeat(True, len(ytrain_n)+1)
            

            ## For each training point, compute the summation needed for adjusting the weights
            ## This corresponds to recursion depths >= 2 in Eq. (16) in Appendix B.2
            for i in range(len(ytrain_n)):
                idx_include_y[i]   = False
                idx_include_y[i-1] = True
                idx_include_x[i]   = False
                idx_include_x[i-1] = True
                summation = self.compute_mfcs_full_cp_scores_weights_ridge_helper(Xaug_n1xp[idx_include_x], ytrain_n[idx_include_y], lmbda, depth_max-1)
                w_n1xy_adjusted[i] = w_n1xy_adjusted[i] * summation
   
            ## Similar weight adjustment for test point, except leave-one-out computation is slightly different 
            ## (i.e., since test pt is left out, certain steps for test pt candidate labels are no longer needed), 
            ## so change flag includes_test=False
            summation = self.compute_mfcs_full_cp_scores_weights_ridge_helper(Xaug_n1xp[:-1], ytrain_n, lmbda, depth_max-1, includes_test=False)
            w_n1xy_adjusted[-1] = w_n1xy_adjusted[-1] * summation
        
        return scoresloo_n1xy, w_n1xy_adjusted

    
    
    @abstractmethod
    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda, depth_max):
        pass
    
    
    
    
    def compute_mfcs_full_cp_scores_weights_ridge_helper(self, Xaug_n1xp_, ytrain_n_, lmbda, depth_max, includes_test: bool = True):
        
        ## Begin by calling subroutine for computing the "one-step" FCS leave-one-out scores and weights
        scoresloo_n1xy_, w_n1xy_ = self.compute_onestep_full_cp_loo_scores_weights_ridge(Xaug_n1xp_, ytrain_n_, lmbda, True, includes_test)
        
        if (depth_max == 1):
            return np.sum(w_n1xy_, axis=0)
        
        else:
            summation = np.repeat(0.0, self.n_y) ## len(summation) = number of y candidates
            idx_include_y = np.repeat(True, len(ytrain_n_)) ## len(idx_include_y) = number of training points
            idx_include_x = np.repeat(True, len(Xaug_n1xp_)) ## len(idx_include_x) = number of training points + 1 test point, if includes_test==True
            
            ## For each row (datapoint), add to the total summation the weights for that point * recursive summation on remaining points
            ## See Eq. (16) in Appendix B.2
            for i in range(len(w_n1xy_)):
                if ((i < len(w_n1xy_)-1) or (includes_test == False)):
                    idx_include_y[i]   = False
                    idx_include_y[i-1] = True
                    idx_include_x[i]   = False
                    idx_include_x[i-1] = True
                    summation += w_n1xy_[i]*self.compute_mfcs_full_cp_scores_weights_ridge_helper(Xaug_n1xp_[idx_include_x], ytrain_n_[idx_include_y], lmbda, depth_max - 1, includes_test)
                else:
                    ## Else == (i==len(w_n1xy_)-1 and includes_test == True): 
                    ## Similar weight adjustment for test point, except leave-one-out computation is slightly different 
                    ## (i.e., since test pt is left out, certain steps for test pt candidate labels are no longer needed), 
                    ## so change flag includes_test=False
                    summation += w_n1xy_[i]*self.compute_mfcs_full_cp_scores_weights_ridge_helper(Xaug_n1xp_[:-1], ytrain_n_, lmbda, depth_max - 1, includes_test=False)

            return summation



    def compute_onestep_full_cp_loo_scores_weights_ridge(self, Xaug_n1xp, ytrain_n, lmbda, compute_weights: bool = True, includes_test: bool = True):
        ## This method is adapted from Fannjiang et al. (2022) "Conformal prediction under feedback covariate shift for biomolecular design";
        ## see that paper's code and the notation in that paper's appendix
        
        
        if (includes_test == True):
            ## This condition is for default case where the last row represents the test point, so len(Xaug_n1xp) = len(ytrain_n) + 1
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
            if compute_weights:
                
                betaiy_nxpxy = C_nxp[:, :, None] + self.ys * An_nxp[:, :, None]
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
            ## This condition is for when every row is a training point (i.e., test point has been removed in 
            ## a previous recursion step), so len(Xaug_n1xp) = len(ytrain_n). Some steps were removed (relative 
            ## to the 'if' condition) that were only necessary for when the test point is included (with its 
            ## grid of candidate labels).
            
            n = ytrain_n.size
            ab_nx1 = np.zeros([n, 1])
            C_nxp = np.zeros([n, self.p])
            for i in range(n):
                # construct A_{-i}
                # Xi_nxp has n-1 rows, since Xaug_n1xp didn't include test point in this condition
                Xi_nxp = np.vstack([Xaug_n1xp[: i], Xaug_n1xp[i + 1 :]]) 
                Ai = get_invcov_dot_xt(Xi_nxp, self.gamma, use_lapack=self.use_lapack)
                
                # compute linear parameterizations of \mu_{-i, y}(X_i)
                yi_ = np.hstack([ytrain_n[: i], ytrain_n[i + 1 :]])  # n - 1 elements
        
                Ci = Ai.dot(yi_) # p elements Ai[:, : -1] changed to Ai
                ai = Ci.dot(Xaug_n1xp[i])  # = Xtrain_nxp[i]

                # store
                ab_nx1[i] = ai 
                C_nxp[i] = Ci

            prediy_nxy = ab_nx1[:, 0][:, None] 

            ## Compute one-step weights w_n1xy
            scoresloo_n1xy = None
            w_nxy = None
            if compute_weights:
               
                betaiy_nxpxy = C_nxp[:, :, None] ##+ self.ys * An_nxp[:, :, None] 
                pred_nxyxu = np.tensordot(betaiy_nxpxy, self.Xuniv_uxp, axes=(1, 1))
                normconst_nxy = np.sum(np.exp(lmbda * pred_nxyxu), axis=2)
                ptrain_n = self.ptrain_fn(Xaug_n1xp)

                ## Each row of these are the n+1 "one-step" weights for a specific candidate label
                w_nxy = np.zeros([n, self.n_y]) ## replaced n+1 with n
                wi_num_nxy = np.exp(lmbda * prediy_nxy)
                
                w_nxy = wi_num_nxy / (ptrain_n[:, None] * normconst_nxy) ## w_n1xy[: -1] replaced with w_n1xy


            return scoresloo_n1xy, w_nxy
            
            



    def compute_confidence_sets(self, Xtrain_nxp, ytrain_n, Xtest_1xp, lmbda, alpha: float = 0.1, depth_max=1, use_is_scores: bool = False):
        if (self.p != Xtrain_nxp.shape[1]):
            raise ValueError('Feature dimension {} differs from provided Xuniv_uxp {}'.format(
                Xtrain_nxp.shape[1], self.Xuniv_uxp.shape))
        Xaug_n1xp = np.vstack([Xtrain_nxp, Xtest_1xp])

        # ===== compute scores and weights =====

        # compute in-sample scores
        scoresis_n1xy = self.get_insample_scores(Xaug_n1xp, ytrain_n) if use_is_scores else None

        # compute LOO scores and likelihood ratios
        scoresloo_n1xy, w_n1xy = self.get_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda, depth_max=depth_max)

        # ===== construct confidence sets =====

        # based on LOO score
        looq_y = get_weighted_quantile(1 - alpha, w_n1xy, scoresloo_n1xy)
        loo_cs = self.ys[scoresloo_n1xy[-1] <= looq_y]

        # based on in-sample score
        is_cs = None
        if use_is_scores:
            isq_y = get_weighted_quantile(1 - alpha, w_n1xy, scoresis_n1xy)
            is_cs = self.ys[scoresis_n1xy[-1] <= isq_y]
        return loo_cs, is_cs


    
    
class FullConformalRidgeExchangeable(FullConformalRidge):
    """
    Class for full conformal with ridge regression, assuming exchangeable data.
    """
    def __init__(self, ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack: bool = True):
        super().__init__(ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack=use_lapack)

    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda, depth_max=0):
        scoresloo_n1xy, _ = self.compute_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda, compute_weights=False)
        # for exchangeble data, equal weights on all data points (no need to compute likelihood ratios in line above)
        w_n1xy = np.ones([Xaug_n1xp.shape[0], self.n_y])
        return scoresloo_n1xy, w_n1xy


class FullConformalRidgeMultistepFeedbackCovariateShift(FullConformalRidge):
    """
    Class for full conformal with ridge regression under multistep feedback covariate shift
    """
    def __init__(self, ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack: bool = True):
        super().__init__(ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack=use_lapack)

    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda, depth_max=1):
        scoresloo_n1xy, w_n1xy = self.compute_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda, compute_weights=True, depth_max=depth_max)
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




class FullConformalBlackBox(ABC):
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

    def compute_confidence_sets(self, Xtrain_nxp, ytrain_n, Xtest_1xp, lmbda,
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

    
    
    
    
    
    
class SplitConformal(ABC):
    """
    Abstract base class for Split Conformal experiments with black-box predictive model.
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

    
    def compute_confidence_sets_design(self, Xtrain_split, Xcal_split, ytrain_split, ycal_split, Xtest_n1xp_split, ytest_n1_split, Xpool_split, w_split_mus_prev_steps, method_names, weight_bounds, t_cal, X_dataset, weight_depth_maxes = [1,2], lmbda = 1/10, bandwidth = 1.0, alpha: float = 0.1, n_initial_all=100, n_dataset = None, replacement=True):
        # , add_to_cal=True
        if (self.p != Xtrain_split.shape[1]):
            raise ValueError('Feature dimension {} differs from provided Xuniv_uxp {}'.format(
                Xtrain_split.shape[1], self.Xuniv_uxp.shape))
#         Xaug_n1xp = np.vstack([Xtrain_nxp, Xtest_1xp])
        Xaug_cal_test_split = np.vstack([Xcal_split, Xtest_n1xp_split])
        # n = ytrain_split.size + ycal_split.size
#         n1 = len(Xaug_n1xp) - n ### Temp removed this 20240111, as chnaged split test points to those queried
        n1 = len(Xtest_n1xp_split)

        ###############################
        # Standard split conformal
        ###############################
        num_cal_curr = len(ycal_split) ## == n_cal_initial + t_cal - 1
        muh_split = self.model.fit(Xtrain_split, ytrain_split)
        muh_split_vals = muh_split.predict(np.r_[Xcal_split,Xtest_n1xp_split])
        resids_split = np.abs(ycal_split-muh_split_vals[:num_cal_curr])
        muh_split_vals_testpoint = muh_split_vals[num_cal_curr:]
        ind_split = (np.ceil((1-alpha)*(num_cal_curr+1))).astype(int)
        
        
        PIs_dict = {'split' : pd.DataFrame(\
                np.c_[muh_split_vals_testpoint - np.sort(resids_split)[ind_split-1], \
                       muh_split_vals_testpoint + np.sort(resids_split)[ind_split-1]],\
                        columns = ['lower','upper'])}
        

        ###############################
        # Weighted split conformal methods for (one-step and multistep) FCS
        ###############################
        
        
        ## Append current ML model (muh function) to list of past models used for querying actively selected calibration points
        ## This list 'w_split_mus_prev_and_curr_steps' will thus allow us to compute the needed query functions p(x | Z_{train}^{(t)})
        w_split_mus_prev_steps.append(deepcopy(self.model))
        w_split_mus_prev_and_curr_steps = w_split_mus_prev_steps


        if (num_cal_curr + 1 != len(muh_split_vals)):
            raise ValueError('num_cal_curr + 1 != len(muh_split_vals); code not yet set up for batch setting.')

        
        ## Compute a matrix of dimension (t_cal, num_cal_curr + 1) of query function values for 
            ## each calibration + test point at each step t \in {1,...t_cal}
        ## Also, compute array of length t_cal for the sum of query function values for all pool points at that step
        w_split_MAT_all_steps = np.zeros((t_cal, num_cal_curr + 1))
        pool_weights_totals_prev_steps = np.zeros(t_cal)
        for i, muh_split_curr in enumerate(w_split_mus_prev_and_curr_steps):

            ## Bound weights with B = weight_bounds[i]
            B = weight_bounds[i]
            
            
            ## Compute (unnormalized) total calibration weights
            y_muh_curr = muh_split_curr.predict(np.r_[Xcal_split,Xtest_n1xp_split])
            # y_muh_curr_minmax_normed = (y_muh_curr - min(y_muh_curr)) / (max(y_muh_curr) - min(y_muh_curr))
            exp_y_muh_curr = np.exp(y_muh_curr * lmbda)
            w_split_MAT_all_steps[i] = np.minimum(exp_y_muh_curr, B*np.ones(len(exp_y_muh_curr)))
            
            
            ## Compute (unnormalized) total pool weights
            y_muh_curr_pool = muh_split_curr.predict(Xpool_split)
            # y_muh_curr_pool_minmax_normed = (y_muh_curr_pool - min(y_muh_curr_pool)) / (max(y_muh_curr_pool) - min(y_muh_curr_pool))
            exp_y_muh_curr_pool = np.exp(y_muh_curr_pool * lmbda)
            pool_weights_totals_prev_steps[i] = np.sum(np.minimum(exp_y_muh_curr_pool, B*np.ones(len(exp_y_muh_curr_pool))))
        
        

        ## Compute the numerator of Eq. (16) in Appendix B.2
        for depth_max in weight_depth_maxes:
            
            depth_max_curr = min(depth_max, t_cal)
            if (replacement):
                ## If sampling with replacement
                Z = pool_weights_totals_prev_steps[-1]
                                
                ## Note: For replacement case, easier to normalize ahead of time by dividing by Z
                split_weights_vec = compute_w_ptest_split_active_replacement(cal_test_vals_mat = w_split_MAT_all_steps/Z, depth_max=depth_max_curr)
                
            else:
                ## Note: Have not finished developing the without replacement case
#                 Z = pool_weights_totals_prev_steps[-1]
#                 ## Note: For replacement case, easier to normalize ahead of time by dividing by Z
                ## Else sampling without replacement
                split_weights_vec = compute_w_ptest_split_active_no_replacement(cal_test_vals_mat = w_split_MAT_all_steps, depth_max = depth_max_curr, pool_weight_arr_curr = pool_weights_totals_prev_steps, n_pool_curr = len(Xpool_split)) #

                
            # print("split_weights_vec shape : ", np.shape(split_weights_vec)) 
            split_weights_vec = split_weights_vec.flatten()
            

            ## Construct empirical distribution of scores (initially unweighted)
            positive_infinity = np.array([float('inf')]) ## conservative adjustment
            unweighted_split_vals = np.concatenate([resids_split, positive_infinity])

            wsplit_quantiles = np.zeros(n1) ## Weighted quantile value for each test point

            ## Compute normalized weights
            weights_normalized_wsplit = np.zeros((num_cal_curr + 1, n1))
            sum_cal_weights = np.sum(split_weights_vec[:num_cal_curr])
            for j in range(0, n1):
                for i in range(0, num_cal_curr + 1):
                    if (i < num_cal_curr):
                        weights_normalized_wsplit[i, j] = split_weights_vec[i] / (sum_cal_weights + split_weights_vec[num_cal_curr + j])
                    else:
                        weights_normalized_wsplit[i, j] = split_weights_vec[num_cal_curr+j] / (sum_cal_weights + split_weights_vec[num_cal_curr + j])

            ## Compute weighted split quantiles for each test point
            for j in range(0, n1):
                wsplit_quantiles[j] = weighted_quantile(unweighted_split_vals, weights_normalized_wsplit[:, j], 1 - alpha)

            
            ## Add results for wsplit with for this estimation depth to full results
            PIs_dict['wsplit_' + str(depth_max)] = pd.DataFrame(np.c_[muh_split_vals_testpoint - wsplit_quantiles, \
                                               muh_split_vals_testpoint + wsplit_quantiles],\
                                               columns = ['lower','upper'])

            
        
        return PIs_dict, w_split_mus_prev_and_curr_steps
    
    
    
    
    
    #### Active learning ####
    def compute_confidence_sets_active(self, Xtrain_split, Xcal_split, ytrain_split, ycal_split, Xtest_n1xp_split, ytest_n1_split, Xpool_split, w_split_mus_prev_steps, exp_vals_pool_list_of_vecs_all_steps, method_names, t_cal, X_dataset, n_cal_initial, alpha_aci_curr, weight_bounds, weight_depth_maxes = [1,2], lmbda = 1/10, bandwidth = 1.0, alpha: float = 0.1, n_initial_all=100, n_dataset = None, replacement=True, record_weights=False):
        # , add_to_cal=True
        if (self.p != Xtrain_split.shape[1]):
            raise ValueError('Feature dimension {} differs from provided Xuniv_uxp {}'.format(
                Xtrain_split.shape[1], self.Xuniv_uxp.shape))
#         Xaug_n1xp = np.vstack([Xtrain_nxp, Xtest_1xp])
        Xaug_cal_test_split = np.vstack([Xcal_split, Xtest_n1xp_split])
        # n = ytrain_n.size
#         n1 = len(Xaug_n1xp) - n ### Temp removed this 20240111, as chnaged split test points to those queried
        n1 = len(Xtest_n1xp_split)
#         n1 = len(ytest_n1_split)

        
        ###############################
        # split conformal
        ###############################
        n_cal = len(ycal_split)
        muh_split = self.model.fit(Xtrain_split, ytrain_split)
        muh_split_vals = muh_split.predict(np.r_[Xcal_split,Xtest_n1xp_split]) # , std_split_vals
        resids_split = np.abs(ycal_split-muh_split_vals[:n_cal])
        muh_split_vals_testpoint = muh_split_vals[n_cal:]
        ind_split = (np.ceil((1-alpha)*(n_cal+1))).astype(int)
        
        
        resids_split_aci = np.concatenate([resids_split,])
        
        PIs_dict = {'split' : pd.DataFrame(\
                np.c_[muh_split_vals_testpoint - np.sort(resids_split)[ind_split-1], \
                       muh_split_vals_testpoint + np.sort(resids_split)[ind_split-1]],\
                        columns = ['lower','upper'])}
        
    
        ###############################
        # Weighted split conformal methods for (one-step and multistep) FCS
        ###############################
        
        
        ## Append current ML model (muh function) to list of past models used for querying actively selected calibration points
        ## This list 'w_split_mus_prev_and_curr_steps' will thus allow us to compute the needed query functions p(x | Z_{train}^{(t)})
        w_split_mus_prev_steps.append(deepcopy(self.model))
        w_split_mus_prev_and_curr_steps = deepcopy(w_split_mus_prev_steps)

        if (n_cal + 1 != len(muh_split_vals)):
            raise ValueError('n_cal + 1 != len(muh_split_vals); code not yet set up for batch setting')

        
        ## Compute a matrix of dimension (t_cal, num_cal_curr + 1) of query function values for 
        ## each calibration + test point at each step t \in {1,...t_cal}
        ## Also, compute array of length t_cal for the sum of query function values for all pool points at that step
        exp_vals_cal_test_MAT_all_steps = np.zeros((t_cal, n_cal + 1))
        exp_vals_pool_sum_all_steps = np.zeros(t_cal)

        
        for i, muh_split_curr in enumerate(w_split_mus_prev_and_curr_steps):
            
            B = weight_bounds[i]
            
            ## Compute (unnormalized) total pool weights
            _, std_pool_muh_curr_ = muh_split_curr.predict(Xpool_split, return_std=True)
            std_pool_muh_curr = get_f_std(std_pool_muh_curr_, muh_split_curr)
            
            var_pool_muh_curr = std_pool_muh_curr**2
            var_pool_muh_curr_minmax_normed = (var_pool_muh_curr) / (max(var_pool_muh_curr) - min(var_pool_muh_curr))
            exp_vars_pool = np.exp(var_pool_muh_curr_minmax_normed * lmbda)

            
            ## Compute (unnormalized) total calibration weights
            _, std_muh_curr_ = muh_split_curr.predict(np.r_[Xcal_split,Xtest_n1xp_split], return_std=True)
            std_muh_curr = get_f_std(std_muh_curr_, muh_split_curr)
            
            var_cal_test_muh_curr = std_muh_curr**2
            var_cal_test_muh_curr_minmax_normed = (var_cal_test_muh_curr)  / (max(var_pool_muh_curr) - min(var_pool_muh_curr))
            
            
            exp_vals_cal_test_MAT_all_steps[i] = np.minimum(np.exp(var_cal_test_muh_curr_minmax_normed * lmbda), B)
            
        
            
            if (np.sum(exp_vars_pool) != np.sum(exp_vals_pool_list_of_vecs_all_steps[i])):
                print("Warning! np.sum(exp_vars_pool) = ", np.sum(exp_vars_pool), "!= np.sum(exp_vals_pool_list_of_vecs_all_steps[i])", np.sum(exp_vals_pool_list_of_vecs_all_steps[i]))
                
            
            
            exp_vals_pool_sum_all_steps[i] = np.sum(np.minimum(exp_vals_pool_list_of_vecs_all_steps[i], B)) 

            

            
        ## Compute the numerator of Eq. (16) in Appendix B.2
        weights_normalized_wsplit_all = [] ## For recording weights, if want to plot how they change over time
        
        for depth_max in weight_depth_maxes:
            
            depth_max_curr = min(depth_max, t_cal)
            
            time_begin_w = time.time()
            
            if (replacement):
                ## If sampling with replacement
                # Z = pool_weights_totals_prev_steps[-1]
                
                
                ## Note: For replacement case, easier to normalize ahead of time by dividing by Z
                split_weights_vec = compute_w_ptest_split_active_replacement(exp_vals_cal_test_MAT_all_steps, depth_max=depth_max_curr)
                


            else:
                ## Note: Have not finished developing the without replacement case
#                 Z = pool_weights_totals_prev_steps[-1]
#                 ## Note: For replacement case, easier to normalize ahead of time by dividing by Z
                ## Else sampling without replacement
                split_weights_vec = compute_w_ptest_split_active_no_replacement(cal_test_vals_mat = w_split_MAT_all_steps, depth_max = depth_max_curr, pool_weight_arr_curr = pool_weights_totals_prev_steps, n_pool_curr = len(Xpool_split)) #

                
                
            # print("Time elapsed for depth ", depth_max_curr, " (min) : ", (time.time() - time_begin_w) / 60)
            
            
            split_weights_vec = split_weights_vec.flatten()

       
            ## Construct empirical distribution of scores (initially unweighted)
            positive_infinity = np.array([float('inf')]) ## Conservative adjustment
            unweighted_split_vals = np.concatenate([resids_split, positive_infinity])

            wsplit_quantiles = np.zeros(n1)

            weights_normalized_wsplit = np.zeros((n_cal + 1, n1))
            sum_cal_weights = np.sum(split_weights_vec[:n_cal])

            for j in range(0, n1):
                for i in range(0, n_cal + 1):
                    if (i < n_cal):
                        weights_normalized_wsplit[i, j] = split_weights_vec[i] / (sum_cal_weights + split_weights_vec[n_cal + j])
                    else:
                        weights_normalized_wsplit[i, j] = split_weights_vec[n_cal+j] / (sum_cal_weights + split_weights_vec[n_cal + j])


            if (record_weights):
                weights_normalized_wsplit_all.append(np.concatenate([sort_both_by_first(unweighted_split_vals[0:n_cal_initial], weights_normalized_wsplit[0:n_cal_initial,0])[1], weights_normalized_wsplit[n_cal_initial:,0]]))
                
    
            for j in range(0, n1):
                wsplit_quantiles[j] = weighted_quantile(unweighted_split_vals, weights_normalized_wsplit[:, j], 1 - alpha)


        
            PIs_dict['wsplit_' + str(depth_max)] = pd.DataFrame(np.c_[muh_split_vals_testpoint - wsplit_quantiles, \
                                               muh_split_vals_testpoint + wsplit_quantiles],\
                                               columns = ['lower','upper'])

        
        
        ###### ACI ######
        
        q_aci = np.quantile(unweighted_split_vals, 1-alpha_aci_curr)
        
                
        PIs_dict['aci'] = pd.DataFrame(np.c_[muh_split_vals_testpoint - q_aci, \
                           muh_split_vals_testpoint + q_aci],\
                            columns = ['lower','upper'])
        
        
        
        return PIs_dict, w_split_mus_prev_steps, weights_normalized_wsplit_all
    
    
    

class FullConformalExchangeable(FullConformalBlackBox):
    """
    Full conformal with black-box predictive model, assuming exchangeable data.
    """
    def __init__(self, model, ptrain_fn, ys, Xuniv_uxp):
        super().__init__(model, ptrain_fn, ys, Xuniv_uxp)

    def get_lrs(self, Xaug_n1xp, yaug_n1, lmbda):
        return np.ones([Xaug_n1xp.shape[0]])


class FullConformalFeedbackCovariateShift(FullConformalBlackBox):
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

    
    

class SplitConformalMFCS(SplitConformal):
    """
    Class for MFCS Split Conformal experiments
    """
    def __init__(self, model, ptrain_fn, Xuniv_uxp):
        super().__init__(model, ptrain_fn, Xuniv_uxp)


        
