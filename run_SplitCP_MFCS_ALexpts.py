import os
import sys
import time
from importlib import reload
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import numpy as np
    
import assay_mfcs
import calibrate_mfcs as cal

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import argparse
from datetime import date

## Added for active learning experiments
from sklearn.neighbors import KernelDensity
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, ConstantKernel, RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn import decomposition

import matplotlib.pyplot as plt


def get_f_std(y_std, gpr):
    params = gpr.kernel_.get_params()

    normalized_noise_var = params["k2__noise_level"]
    y_train_var = gpr._y_train_std ** 2

    y_pred_var = y_std ** 2
    f_pred_var = y_pred_var - (y_train_var * normalized_noise_var)
    f_std = np.sqrt(f_pred_var)
    return f_std


def get_PCA(x):
    pca = decomposition.PCA(n_components=1)
    pca.fit(x)
    x_pca = pca.transform(x)
    return x_pca
    

def noisy_measurements(idx_y, Y_vals_all, errs, noise_magnitude, seed: int = None):
    """
    Given indices of sequences, return noisy measurements (using estimated measurement noise SD).

    :param idx_y: indices for y values
    :param muh: predictor function whose predictions are used for estimating measurement noise magnitude
    :param seed: int, random seed
    :return: numpy array of noisy measurements corresponding to provided sequence indices
    """
#     np.random.seed(seed)
    noisy_n = np.array([np.random.normal(loc=Y_vals_all[i], scale=noise_magnitude*errs[i]) for i in idx_y])
    # enforce non-negative measurement since enrichment scores are always non-negative
    return noisy_n

def compute_errors_for_measurements(X_vals, Y_vals):
    """
    Given indices of sequences, return noisy measurements (using estimated measurement noise SD).

    :param Y_vals: Y values
    :param muh: predictor function whose predictions are used for estimating measurement noise magnitude
    :param seed: int, random seed
    :return: numpy array of noisy measurements corresponding to provided sequence indices
    """

    krr = KernelRidge(alpha=1.0)
    krr.fit(X_vals, Y_vals)
    return abs(Y_vals - krr.predict(X_vals))


            
def compute_weight_bound_binary_search(prob_bound_inf, query_vals_cal, query_vals_pool_sorted, start, end, alpha=0.1):
    """
    :param prob_bound_inf   : Largest acceptable probability of infinite interval widths (0,1); e.g., 1.0 is unbounded, 0.0 is bounded query function (Appendix C)
    :param query_vals_cal  : Unnormalized, unbounded weights (exponentiated tilt on predicted fitness values for calibration data)
    :param query_vals_pool_sorted : Unnormalized, unbounded weights (exponentiated tilt on predicted fitness values for pool) already sorted
    Compute B : largest bound for which the probability of an infinite interval width is maintained below 'prob_bound_inf'
    """
    mid = int((end + start) / 2)
    
    b = query_vals_pool_sorted[mid]
    
    total_weight_cal = np.sum(np.minimum(query_vals_cal, b))
    unnormalized_weights_pool = np.minimum(query_vals_pool_sorted, b)
    inds_infty_pool = np.where(unnormalized_weights_pool / (total_weight_cal + unnormalized_weights_pool) >= alpha)[0]

    if (start + 1 == end):
        return start, query_vals_pool_sorted[start]
                
    if (np.sum(unnormalized_weights_pool[inds_infty_pool]) / np.sum(unnormalized_weights_pool) <= prob_bound_inf):
        return compute_weight_bound_binary_search(prob_bound_inf, query_vals_cal, query_vals_pool_sorted, start=mid, end=end, alpha=alpha)

    else:
        return compute_weight_bound_binary_search(prob_bound_inf, query_vals_cal, query_vals_pool_sorted, start=start, end=mid, alpha=alpha)
    
    
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run multistep FCS experiments.')
    
    parser.add_argument('--dataset', type=str, default='airfoil', help='Dataset name.')
    parser.add_argument('--n_initial_all', type=int, help='Initial number of training points', required = True)
    parser.add_argument('--lmbdas', nargs='+', help='Values of lmbda to try', required = True)
    parser.add_argument('--n_seed', type=int, default=500, help='Number of trials')
    parser.add_argument('--n_steps', type=int, default=10, help='Number of MFCS steps (active learning iterations)')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha value corresponding to 1-alpha target coverage')
#     parser.add_argument('--K_vals', nargs='+', help='Values of K to try', required = True)
    parser.add_argument('--muh', type=str, default='GP', help='Muh predictor.')
    parser.add_argument('--seed_initial', type=int, default=0, help='Initial seed')
    parser.add_argument('--replacement', type=bool, default=True, help='Sample queries with or without replacement')
    parser.add_argument('--n_queries_ann', type=int, default=1, help='Number of queries')
    parser.add_argument('--n_val', type=int, default=250, help='Number of validation points for MSE')
    parser.add_argument('--p_split_train', type=float, default=0.5, help='Initial proportion of data that goes to training')
    parser.add_argument('--add_to_train_split_prob', type=float, default=0.5, help='Probability of adding a query point to training')
    parser.add_argument('--weight_depth_maxes', nargs='+', default=[1,2,3,4], help='Estimation depths of weight adjustments to include')
    parser.add_argument('--kernel_str', type=str, default='dot_white', help='Kernel type for GP')
    parser.add_argument('--noise_magnitude', type=float, default=0.1, help='Positive value controlling noise magnitude. 1 means noise is one std')
    parser.add_argument('--initial_sampling_bias', type=float, default=1.0, help='Expontential tilting of how training & calibration data are sampled from the full dataset, w.r.t. first PCA of the data.')
    parser.add_argument('--save_weight_computations', type=bool, default=False, help='Bool about whether to save weights')
    parser.add_argument('--noise_level', type=float, default=1.0, help='White kernel noise level')
    parser.add_argument('--sigma_0', type=float, default=1.0, help='Dot product kernel noise level')
    parser.add_argument('--aci_step_size', type=float, default=0.005, help='ACI step size gamma')
    parser.add_argument('--prob_bound_inf', type=float, default=1.0, help='prob_bound_inf : max acceptable probability of infinite interval width. prob_bound_inf=1.0 --> unbounded query function; prob_bound_inf=0.0 --> bounded query function (Appendix C)')
    parser.add_argument('--record_weights', type=bool, default=False, help='Bool indicating whether to record the calibration + test point weights (takes extra storage and time)')

    ## python run_SplitCP_MultistepFCS_expts.py --dataset airfoil --n_initial_all 48 --lmbdas 10 --n_seed 200 --p_split_train 0.6 
    
    args = parser.parse_args()
    dataset = args.dataset
    n_initial_all = args.n_initial_all
    lmbdas = [float(lmbda) for lmbda in args.lmbdas]
    lmbdas_str = ''.join(str(l) for l in lmbdas)
    n_seed = args.n_seed
    alpha = args.alpha
    muh = args.muh
    n_steps = args.n_steps
    n_queries_ann = args.n_queries_ann
    n_test = n_queries_ann
    replacement = args.replacement
    add_to_train_split_prob = args.add_to_train_split_prob
    weight_depth_maxes = [int(dep) for dep in args.weight_depth_maxes]
    seed_initial = args.seed_initial
    n_val = args.n_val
    p_split_train = args.p_split_train
    str_dep_maxes = ''
    for d in weight_depth_maxes:
        str_dep_maxes += str(d)
    kernel_str = args.kernel_str
    noise_magnitude = args.noise_magnitude
    initial_sampling_bias = args.initial_sampling_bias
    save_weight_computations = args.save_weight_computations
    noise_level = args.noise_level
    sigma_0 = args.sigma_0
    aci_step_size = args.aci_step_size
    prob_bound_inf = args.prob_bound_inf
    record_weights = args.record_weights
    
    
    start_time = time.time()


    if (muh == 'GP'):
        if(kernel_str == 'RBF'):
            kernel = RBF()
        elif (kernel_str == 'constant'):
            kernel = ConstantKernel()
        else:
            kernel = DotProduct(sigma_0=sigma_0) + WhiteKernel(noise_level=noise_level)
        muh_fun = GaussianProcessRegressor(kernel=kernel,random_state=0)

    method_names = ['split']
    for d in weight_depth_maxes:
        method_names.append('wsplit_' + str(d))
        
    method_names.append('aci')
    
    print("Running with methods : ", method_names)
    
    
    
    print('Running with '+ dataset + '_' + muh + '_nInitial' + str(n_initial_all) + '_steps' + str(n_steps) + '_nseed' + str(n_seed) + '_lmbda' + lmbdas_str + '_wDepths' + str_dep_maxes + '_PIs_addTrainProb'+str(add_to_train_split_prob)+'_replace'+str(replacement)+'_kernel'+kernel_str +'pcaExpSampling'+str(initial_sampling_bias) + str(aci_step_size)+ '_probBoundInf' + str(prob_bound_inf))
    


    n_train_proper_initial = int(np.floor(n_initial_all * p_split_train))
    n_cal_initial = n_initial_all - n_train_proper_initial
    
    
    if (record_weights):
        results_by_seed = pd.DataFrame(columns = np.concatenate([['seed', 'step', 't_cal', 'dataset', 'muh_fun','method','coverage','width', 'MSE', 'muh_test', 'y_test', 'B', 'prop_B', 'alpha_aci'], ['w_cal' + str(i) for i in range(0, n_cal_initial+n_steps-1)], ['w_test']]))
    else:
        results_by_seed = pd.DataFrame(columns = np.concatenate([['seed', 'step', 't_cal', 'dataset', 'muh_fun','method','coverage','width', 'MSE', 'muh_test', 'y_test', 'B', 'prop_B', 'alpha_aci']]))

    

    ## Likelihood under training input distribution, p_X in paper (uniform distribution in protein design experiments).
    ## Due to initializing with IID *uniform random* samples, the following two lines will yield equivalent results, but both
    ## are provided to correspond to different factorizations given in Appendix B.1 (direct likelihood factorization approach)
    ## and Appendix B.3 (likelihood-*ratio*-based factorization) respectively.
    ptrain_fn = lambda x: np.ones([x.shape[0]]) ## Trivial denominator (=1) for direct likelihood factorization (as in Appendix B.1 in paper)
    # ptrain_fn = lambda x: (1.0 / np.power(2, 13)) * np.ones([x.shape[0]]) ## Likelihood-ratio based factorization (see Appendix B.3 in paper)



    # Read dataset
    if (dataset == 'airfoil'):
        airfoil = pd.read_csv(os.getcwd().removesuffix('bash_scripts') + '/datasets/airfoil/airfoil.txt', sep = '\t', header=None)
        airfoil.columns = ["Frequency","Angle","Chord","Velocity","Suction","Sound"]
        X_airfoil = airfoil.iloc[:, 0:5].values
        X_airfoil[:, 0] = np.log(X_airfoil[:, 0])
        X_airfoil[:, 4] = np.log(X_airfoil[:, 4])
        Y_airfoil = airfoil.iloc[:, 5].values
        n_airfoil = len(Y_airfoil)
        errs_airfoil = compute_errors_for_measurements(X_airfoil, Y_airfoil)


    elif (dataset == 'wine'):
        winequality_red = pd.read_csv(os.getcwd().removesuffix('bash_scripts') + '/datasets/wine/winequality-red.csv', sep=';')
        X_wine = winequality_red.iloc[:, 0:11].values
        Y_wine = winequality_red.iloc[:, 11].values
        n_wine = len(Y_wine)
        errs_wine = compute_errors_for_measurements(X_wine, Y_wine)

        print("X_wine shape : ", X_wine.shape)


    elif (dataset == 'communities'):
        # UCI Communities and Crime Data Set
        # download from:
        # http://archive.ics.uci.edu/ml/datasets/communities+and+crime
        communities_data = np.loadtxt(os.getcwd().removesuffix('bash_scripts') + '/datasets/communities/communities.data',delimiter=',',dtype=str)
        # remove categorical predictors
        communities_data = np.delete(communities_data,np.arange(5),1)
        # remove predictors with missing values
        communities_data = np.delete(communities_data,\
                    np.argwhere((communities_data=='?').sum(0)>0).reshape(-1),1)
        communities_data = communities_data.astype(float)
        X_communities = communities_data[:,:-1]
        Y_communities = communities_data[:,-1]
        n_communities = len(Y_communities)
        errs_communities = compute_errors_for_measurements(X_communities, Y_communities)
        print("X_communities shape : ", X_communities.shape)

    elif (dataset == 'meps'):
        meps_data = np.loadtxt(os.getcwd().removesuffix('bash_scripts') + '/datasets/meps/meps_data.txt')
        X_meps = meps_data[:,:-1]
        Y_meps = meps_data[:,-1]
        n_meps = len(Y_meps)
        errs_meps = compute_errors_for_measurements(X_meps, Y_meps)
        print("X_meps shape : ", X_meps.shape)

    elif (dataset == 'blog'):
        blog_data = np.loadtxt(os.getcwd().removesuffix('bash_scripts') + '/datasets/blog/blogData_train.csv',delimiter=',')
        X_blog = blog_data[:,:-1]
        Y_blog = np.log(1+blog_data[:,-1])
        n_blog = len(Y_blog)
        errs_blog = compute_errors_for_measurements(X_blog, Y_blog)



    X_all = eval('X_'+dataset)
    all_inds = np.arange(eval('n_'+dataset))
    
    split_mfcs_AL = cal.SplitConformalMFCS(muh_fun, ptrain_fn, X_all)

    
    ## Loop for repeating experiment with different lmbdas (shift magnitudes)
    for l, lmbda in enumerate(lmbdas):
        

        for seed in range(seed_initial, seed_initial + n_seed):
            print("seed : ", seed)
            Y_all = noisy_measurements(all_inds, eval('Y_'+dataset), eval('errs_'+dataset), noise_magnitude, seed)

            ## Note: Validation set won't change, train and pool will
            np.random.seed(seed)
            val_inds = list(np.random.choice(eval('n_'+dataset),n_val,replace=False))
            non_val_inds = np.setdiff1d(all_inds, val_inds)
            X_nonval = X_all[non_val_inds]

            ## Bias initial sampling of training and calibration data to simulate selection bias in active learning
            X_nonval_pca = get_PCA(X_nonval).flatten()
            X_nonval_pca_minmax = (X_nonval_pca - min(X_nonval_pca)) / (max(X_nonval_pca) - min(X_nonval_pca))
            X_nonval_pca_minmax_nonvals_exp = np.exp(X_nonval_pca_minmax * initial_sampling_bias)
            X_nonval_pca_minmax_nonvals_exp_normed = X_nonval_pca_minmax_nonvals_exp / np.sum(X_nonval_pca_minmax_nonvals_exp)

            train_inds = list(np.random.choice(non_val_inds, n_initial_all, replace=replacement, p=X_nonval_pca_minmax_nonvals_exp_normed))

            ## Pool inds are those not in training or validation data.
            pool_inds = list(np.setdiff1d(np.setdiff1d(np.arange(eval('n_'+dataset)),train_inds), val_inds)) 

            ## Create validation set (won't change)
            Xval = eval('X_'+dataset)[val_inds]
            yval = Y_all[val_inds]

            idx_split = list(np.random.permutation(train_inds))
            train_inds_split, cal_inds_split = list(idx_split[:n_train_proper_initial]), list(idx_split[n_train_proper_initial:])

            ## Note: Calibration set for split won't change
            Xtrain_split = eval('X_'+dataset)[train_inds_split]
            ytrain_split = Y_all[train_inds_split]

            Xcal_split = eval('X_'+dataset)[cal_inds_split]
            ycal_split = Y_all[cal_inds_split]

            ## Pool inds for split are initially the same but will be different later
            pool_inds_split = list(np.setdiff1d(np.setdiff1d(np.arange(eval('n_'+dataset)),train_inds), val_inds))

            ## Initialize train and pool data for sample splitting (will change)
            Xpool_split = eval('X_'+dataset)[pool_inds_split]
            ypool_split = Y_all[pool_inds_split]


            ## Iterate through active learning steps
            exp_vals_pool_list_of_vecs_all_steps = []
            w_split_mus_prev_steps = []

            ## t_cal = (num points in calibration set actively selected) + 1
            t_cal = 1

            ## Initialize alpha_aci_curr to alpha
            alpha_aci_curr = alpha
            weight_bounds = []

            for step in range(n_steps):

                ## 1. Fit Gaussian process regression and use it to select queries from pool
                gpr_split = muh_fun.fit(Xtrain_split, ytrain_split)

                ## 2. Compute unnormalized weights for pool
                y_preds_pool_split, std_preds_pool_split_ = gpr_split.predict(Xpool_split, return_std=True) 
                std_preds_pool_split = get_f_std(std_preds_pool_split_, gpr_split)

                var_preds_pool_split = std_preds_pool_split**2
                var_preds_pool_split_minmax_normed = (var_preds_pool_split) / (max(var_preds_pool_split) - min(var_preds_pool_split))
                exp_var_preds_pool_split = np.exp(var_preds_pool_split_minmax_normed * lmbda)

                exp_vals_pool_list_of_vecs_all_steps.append(exp_var_preds_pool_split) 

                exp_var_preds_pool_split_sorted = np.sort(exp_var_preds_pool_split)

                
                ## 3. Compute unnormalized weights for calibration data
                ycal_preds_split, std_preds_cal_split_ = gpr_split.predict(Xcal_split, return_std=True)
                std_preds_cal_split = get_f_std(std_preds_cal_split_, gpr_split)

                var_preds_cal_split = std_preds_cal_split**2
                var_preds_cal_split_minmax_normed = (var_preds_cal_split) / (max(var_preds_pool_split) - min(var_preds_pool_split))
                exp_var_preds_cal_split = np.exp(var_preds_cal_split_minmax_normed * lmbda)


                ## 4. Compute bound on weight
                i_B, B = compute_weight_bound_binary_search(prob_bound_inf, exp_var_preds_cal_split, exp_var_preds_pool_split_sorted, start=0, end=len(exp_var_preds_pool_split_sorted)-1, alpha=alpha)
                weight_bounds.append(B)


                ## 5. Compute normalized pool weights and use those for sampling a query point
                probs_pool_split = np.minimum(exp_var_preds_pool_split, B) / np.sum(np.minimum(exp_var_preds_pool_split, B))


                query_ann_inds_split = list(np.random.choice(pool_inds_split, n_queries_ann, replace=replacement, p=probs_pool_split)) 

                Xtest_split = eval('X_'+dataset)[query_ann_inds_split]

                ytest_split = noisy_measurements(query_ann_inds_split, eval('Y_'+dataset), eval('errs_'+dataset), noise_magnitude, seed)

                ## 6. For later MSE calculation: Compute predictions on validation data
                y_preds_val_split, std_preds_val_split_ = gpr_split.predict(Xval, return_std=True)
                std_preds_val_split = get_f_std(std_preds_val_split_, gpr_split)

                ytest_preds_split = gpr_split.predict(Xtest_split)

                
                ## 7. Only if sampling without replacement, remove queried point from pool
                ## (note, code not tested on without replacement case yet)
                if (not replacement):

                    pool_inds_split = list(set(pool_inds_split) - set(query_ann_inds_split))
                    Xpool_split = eval('X_'+dataset)[pool_inds_split]



                PIs, w_split_mus_prev_steps, weights_normalized_wsplit_all = split_mfcs_AL.compute_confidence_sets_active(Xtrain_split, Xcal_split, ytrain_split, ycal_split, Xtest_split, ytest_split, Xpool_split, w_split_mus_prev_steps, exp_vals_pool_list_of_vecs_all_steps, method_names, t_cal=t_cal, X_dataset = eval('X_'+dataset), n_cal_initial = n_cal_initial, alpha_aci_curr = alpha_aci_curr, weight_bounds = weight_bounds, weight_depth_maxes=weight_depth_maxes, lmbda = lmbda, bandwidth = 1.0, alpha=alpha, n_initial_all = n_initial_all, n_dataset = eval('n_'+dataset), replacement = replacement, record_weights = record_weights)

                MSE_split = np.mean((y_preds_val_split - yval)**2)


                for m_i, method in enumerate(method_names):

                    coverage_by_seed = ((PIs[method]['lower'] <= ytest_split)&(PIs[method]['upper'] >= ytest_split)).mean()
                    muh_test_by_seed = ytest_preds_split.mean()
                    coverage_all = ((PIs[method]['lower'] <= ytest_split)&(PIs[method]['upper'] >= ytest_split))
                    muh_test_all = ytest_preds_split
                    ytest_method = ytest_split
                    MSE = MSE_split

                    width_by_seed = (PIs[method]['upper'] - PIs[method]['lower']).median()
                    width_all = (PIs[method]['upper'] - PIs[method]['lower'])

                    if (m_i == 0 or method == 'aci'):
                        
                        if (record_weights):
                            cal_test_weights = np.concatenate([np.repeat(1 / (n_cal_initial + t_cal), \
                                                                     n_cal_initial + t_cal - 1), \
                                                           np.repeat(0, n_steps-t_cal), \
                                                           [1/(n_cal_initial + t_cal)]])


                    else:
                        if (record_weights):
                            cal_test_weights = np.concatenate([weights_normalized_wsplit_all[m_i-1][:-1], \
                                                               np.repeat(0, n_steps-t_cal), [weights_normalized_wsplit_all[m_i-1][-1]]])


                    prop_B = i_B / len(pool_inds_split)
                    
                    if (record_weights):
                        results_by_seed.loc[len(results_by_seed)]=np.concatenate([[seed,step,t_cal-1, dataset, muh, method,coverage_by_seed,width_by_seed,MSE,\
                                                                                   muh_test_all[0], ytest_method[0], B, prop_B, alpha_aci_curr], cal_test_weights])
                    else:
                        results_by_seed.loc[len(results_by_seed)]=np.concatenate([[seed,step,t_cal-1, dataset, muh, method,coverage_by_seed,width_by_seed,MSE,\
                                                                                   muh_test_all[0], ytest_method[0], B, prop_B, alpha_aci_curr]])

                    if (method == 'aci'):

                        if (PIs[method]['lower'].values[0] <= ytest_split[0] and PIs[method]['upper'].values[0] >= ytest_split[0]):
                            alpha_aci_curr = min(alpha_aci_curr + aci_step_size * (alpha),1)
                        else:
                            alpha_aci_curr = max(alpha_aci_curr + aci_step_size * (alpha - 1),0)


                ## Add point that was queried for annotation to the training or calibration data 
                U = np.random.uniform()
                if (U <= add_to_train_split_prob):
                    ### Update train data for sample splitting

                    for q_ann in query_ann_inds_split:
                        train_inds_split.append(q_ann) ## Add queried samples to training set

                    Xtrain_split = eval('X_'+dataset)[train_inds_split]
                    ytrain_split = np.concatenate((ytrain_split, ytest_split))

                    ## If added point to training data, then do not record this weight function or the exp pool weights
                    w_split_mus_prev_steps = w_split_mus_prev_steps[:-1]
                    exp_vals_pool_list_of_vecs_all_steps = exp_vals_pool_list_of_vecs_all_steps[:-1]
                    weight_bounds = weight_bounds[:-1]


                else:
                    ### Update calibration data for sample splitting
                    for q_ann in query_ann_inds_split:
                        cal_inds_split.append(q_ann) ## Add queried samples to training set

                    Xcal_split = eval('X_'+dataset)[cal_inds_split]
                    ycal_split = np.concatenate((ycal_split, ytest_split))

                    ## Incrememnt the number of actively selected calibration points
                    t_cal += 1


                
                    
        if (((seed+1) % 50) == 0):
            results_by_seed.to_csv(os.getcwd().removesuffix('bash_scripts') + '/results/'+ str(date.today()) + '_ALExpts_' + dataset + '_' + muh + '_nInitial' + str(n_initial_all) + '_steps' + str(n_steps) + '_nseed' + str(n_seed) + '_lmbda' + str(lmbda) + '_wDepths' + str_dep_maxes + '_propTrainInit'+ str(p_split_train) +'_addTrainProb'+str(add_to_train_split_prob)+'_noise'+str(noise_magnitude)+'_pcaExpSampling'+str(initial_sampling_bias)+ '_GPRnoise' + str(noise_level)  + '_sigDotProd' + str(sigma_0)+ '_aciStepSize' + str(aci_step_size)+ '_probBoundInf' + str(prob_bound_inf) + '.csv',index=False)

    end_time = time.time()

    print("Total time (minutes) : ", (end_time - start_time)/60)

    results_by_seed.to_csv(os.getcwd().removesuffix('bash_scripts') + '/results/'+ str(date.today()) + '_ALExpts_' + dataset + '_' + muh + '_nInitial' + str(n_initial_all) + '_steps' + str(n_steps) + '_nseed' + str(n_seed) + '_lmbda' + str(lmbda) + '_wDepths' + str_dep_maxes + '_PIs_propTrainInit'+ str(p_split_train) +'_addTrainProb'+str(add_to_train_split_prob)+'_noise'+str(noise_magnitude)+'_pcaExpSampling'+str(initial_sampling_bias)+ '_GPRnoise' + str(noise_level) + '_sigDotProd' + str(sigma_0)+ '_aciStepSize' + str(aci_step_size)+'_probBoundInf' + str(prob_bound_inf) +'.csv',index=False)
