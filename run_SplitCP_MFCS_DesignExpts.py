import os
import sys
import time
from importlib import reload
module_path = os.path.abspath(os.path.join('./fluorescence'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import numpy as np
    
import assay_mfcs as assay
import calibrate_mfcs as cal

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import argparse
from datetime import date


import matplotlib.pyplot as plt


def noisy_measurements(idx_y, Y_vals_all, errs, noise_magnitude, seed: int = None):
    """
    Given indices of sequences, return noisy measurements (using estimated measurement noise SD).

    :param idx_y: indices for y values
    :param muh: predictor function whose predictions are used for estimating measurement noise magnitude
    :param seed: int, random seed
    :return: numpy array of noisy measurements corresponding to provided sequence indices
    """
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
    parser.add_argument('--n_initial_all', type=int, help='Initial number of training + calibration points', required = True)
    parser.add_argument('--lmbdas', nargs='+', help='Values of lmbda to try', required = True)
    parser.add_argument('--n_seed', type=int, default=500, help='Number of trials')
    parser.add_argument('--n_steps', type=int, default=10, help='Number of MFCS steps (active learning iterations)')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha value corresponding to 1-alpha target coverage')
    parser.add_argument('--muh', type=str, default='NN', help='Muh predictor.')
    parser.add_argument('--seed_initial', type=int, default=0, help='Initial seed')
    parser.add_argument('--replacement', type=bool, default=True, help='Sample queries with or without replacement')
    parser.add_argument('--n_queries_ann', type=int, default=1, help='Number of queries')
    parser.add_argument('--n_val', type=int, default=250, help='Number of validation points for MSE')
    parser.add_argument('--p_split_train', type=float, default=0.5, help='Initial proportion of data that goes to training')
    parser.add_argument('--add_to_train_split_prob', type=float, default=0.5, help='Probability of adding a query point to training')
    parser.add_argument('--weight_depth_maxes', nargs='+', default=[1,2,3], help='Estimation depths of weight adjustments to include')
    parser.add_argument('--kernel_str', type=str, default='dot_white', help='Kernel type for GP')
    parser.add_argument('--noise_magnitude', type=float, default=0.1, help='Positive value controlling noise magnitude. 1 means noise is one std')
    parser.add_argument('--prob_bound_inf', type=float, default=1.0, help='prob_bound_inf : max acceptable probability of infinite interval width. prob_bound_inf=1.0 --> unbounded query function; prob_bound_inf=0.0 --> bounded query function (Appendix C)')

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
    prob_bound_inf = args.prob_bound_inf
    
    start_time = time.time()



    if (muh == 'GP'):
        if(kernel_str == 'RBF'):
            kernel = RBF()
        elif (kernel_str == 'constant'):
            kernel = ConstantKernel()
        else:
            kernel = DotProduct() + WhiteKernel()
        muh_fun = GaussianProcessRegressor(kernel=kernel,random_state=0)
    elif (muh == 'NN'):
        muh_fun = MLPRegressor(solver='lbfgs',activation='logistic', max_iter=500)
    elif (muh == 'RF'):
        muh_fun = RandomForestRegressor(n_estimators=20,criterion='absolute_error')

    method_names = ['split']
    for d in weight_depth_maxes:
        method_names.append('wsplit_' + str(d))
        
    print("Running with methods : ", method_names)
    
        
    print('Running with '+ dataset + '_' + muh + '_nInitial' + str(n_initial_all) + '_steps' + str(n_steps) + '_nseed' + str(n_seed) + '_lmbda' + lmbdas_str + '_wDepths' + str_dep_maxes + '_propTraini'+ str(p_split_train) +'_addTrainProb'+str(add_to_train_split_prob)+'_noise'+str(noise_magnitude)+'_probBoundInf' + str(prob_bound_inf))



    results_by_seed = pd.DataFrame(columns = ['seed', 'step', 'dataset', 'muh_fun','method','coverage','width', 'MSE'])
    results_all = pd.DataFrame(columns = ['seed','step', 'test_pt', 'dataset','muh_fun','method','coverage','width', 'muh_test', 'y_test'])

    ## Likelihood under training input distribution, p_X in paper (uniform distribution in protein design experiments).
    ## Due to initializing with IID *uniform random* samples, the following two lines will yield equivalent results, but both
    ## are provided to correspond to different factorizations given in Appendix B.1 (direct likelihood factorization approach)
    ## and Appendix B.3 (likelihood-*ratio*-based factorization) respectively.
    ptrain_fn = lambda x: np.ones([x.shape[0]]) ## Trivial denominator (=1) for direct likelihood factorization (as in Appendix B.1 in paper)
    # ptrain_fn = lambda x: (1.0 / np.power(2, 13)) * np.ones([x.shape[0]]) ## Likelihood-ratio based factorization (see Appendix B.3 in paper)
    

    if (dataset == 'red_protein'):

        #######################
        #### This section added for protein design 
        #######################
        fitness_str = 'red'
        reload(cal)
        reload(assay)
        #     n_trains = [96, 192, 384]    # 96, 192,          # number of training points
        ntrain2reg = {96: 10, 192: 1, 384: 1} # ridge regularization strength (gamma in code and paper)
        #     n_seed = 1000                       # number of random trials  ## Drew changed this from 2000
        #     lmbdas = [0, 2, 4, 6]  # 0, 2,                # lambda, inverse temperature
        order = 2                             # complexity of features. 1 encodes the AA at each site,
                                              # 2 the AAs at each pair of sites,
                                              # 3 the AAs at each set of 3 sites, etc.

        red_protein_data = assay.PoelwijkData(fitness_str, order=order)
        X_red_protein = red_protein_data.X_nxp
        n_red_protein = len(X_red_protein)
        Y_red_protein = red_protein_data.get_measurements(range(n_red_protein))


        #######################

    elif (dataset == 'blue_protein'):

        #######################
        #### This section added for protein design 
        #######################
        fitness_str = 'blue'
        reload(cal)
        reload(assay)
        #     n_trains = [96, 192, 384]    # 96, 192,          # number of training points
        ntrain2reg = {96: 10, 192: 1, 384: 1} # ridge regularization strength (gamma in code and paper)
        #     n_seed = 1000                       # number of random trials  ## Drew changed this from 2000
        #     lmbdas = [0, 2, 4, 6]  # 0, 2,                # lambda, inverse temperature
        order = 2                             # complexity of features. 1 encodes the AA at each site,
                                              # 2 the AAs at each pair of sites,
                                              # 3 the AAs at each set of 3 sites, etc.

        blue_protein_data = assay.PoelwijkData(fitness_str, order=order)
        X_blue_protein = blue_protein_data.X_nxp
        n_blue_protein = len(X_blue_protein)
        Y_blue_protein = blue_protein_data.get_measurements(range(n_blue_protein))


    X_all = eval('X_'+dataset)
    all_inds = np.arange(eval('n_'+dataset))

    split_mfcs_design = cal.SplitConformalMFCS(muh_fun, ptrain_fn, X_all)


    ## Loop for repeating experiment with different lmbdas (shift magnitudes)
    for l, lmbda in enumerate(lmbdas):
        
        ## Loop for repeating experiment with different random seeds (which changes the training sample)
        for seed in range(seed_initial, seed_initial + n_seed):
            print("seed : ", seed)
            Y_all = eval(dataset + '_data').get_measurements(range(eval('n_'+dataset)))

            np.random.seed(seed)

            ## Note: Validation set won't change, train and pool will
            val_inds = list(np.random.choice(eval('n_'+dataset),n_val,replace=replacement))
            train_and_cal_inds = list(np.random.choice(np.setdiff1d(np.arange(eval('n_'+dataset)),val_inds), n_initial_all, replace=replacement))

            ## Create validation set (won't change)
            Xval = eval('X_'+dataset)[val_inds]
            yval = Y_all[val_inds]


            ## Sample splitting indices
            idx_split = list(np.random.permutation(train_and_cal_inds))

            n_half_initial = int(np.floor(n_initial_all * p_split_train))
            train_inds_split, cal_inds_split = list(idx_split[:n_half_initial]), list(idx_split[n_half_initial:])

            ## Note: Calibration set for split won't change
            Xtrain_split = eval('X_'+dataset)[train_inds_split]
            ytrain_split = Y_all[train_inds_split]

            Xcal_split = eval('X_'+dataset)[cal_inds_split]
            ycal_split = Y_all[cal_inds_split]

            ## For protein design experiments, the pool is the full dataset (sampling with replacement)
            pool_inds_split = all_inds
            # pool_inds_split = list(np.setdiff1d(np.setdiff1d(np.arange(eval('n_'+dataset)), train_and_cal_inds), val_inds))

            ## Initialize train and pool data for sample splitting (will change)
            Xpool_split = eval('X_'+dataset)[pool_inds_split]
            ypool_split = Y_all[pool_inds_split]

            ## Iterate through active learning steps
            w_split_mus_prev_steps = []

            ## t_cal = (num points in calibration set actively selected) + 1
            t_cal = 1
            weight_bounds = []
            
            
            ## Loop for each step in the multistep design process (in each step >= 2, one point is queried by the ML model, 
            ## a prediction set is computed for the point, and it is then subsequently labeled and added to training).
            for step in range(n_steps):

                ## 1. Fit Gaussian process regression and use it to select queries from pool
                muh_predictor_split = muh_fun.fit(Xtrain_split, ytrain_split)

                ## 2. Compute unnormalized weights for pool
                y_preds_pool_split = muh_predictor_split.predict(X_all) ## 20240117: Changed from Xpool_split to X_all
                exp_y_preds_pool_split = np.exp(y_preds_pool_split * lmbda)
                exp_y_preds_pool_split_sorted = np.sort(exp_y_preds_pool_split)

                ## 3. For later: Save predicted fitness values for the calibration data at this step
                ycal_preds_split = muh_predictor_split.predict(Xcal_split)        
                exp_ycal_preds_split = np.exp(ycal_preds_split * lmbda)

                ## 4. Compute bound on weight
            
                i_B, B = compute_weight_bound_binary_search(prob_bound_inf = prob_bound_inf, query_vals_cal = exp_ycal_preds_split, query_vals_pool_sorted = exp_y_preds_pool_split_sorted, start=0, end=len(exp_y_preds_pool_split_sorted)-1, alpha=alpha)
                
                weight_bounds.append(B)


                ## 5. Sample query points from the pool
                probs_pool_split = np.minimum(exp_y_preds_pool_split, B)/np.sum(np.minimum(exp_y_preds_pool_split, B))
                query_ann_inds_split = list(np.random.choice(pool_inds_split, n_queries_ann, replace=replacement, p=probs_pool_split)) 

                Xtest_split = eval('X_'+dataset)[query_ann_inds_split]
                ytest_split = eval(dataset + '_data').get_measurements(query_ann_inds_split)


                ## 6. (This is only used for active learning) For later MSE calculation: Compute predictions on validation data
                y_preds_val_split = muh_predictor_split.predict(Xval)
                ytest_preds_split = muh_predictor_split.predict(Xtest_split)

                ## If not sampling with replacement, then remove the queried points from the pool 
                ## (Code not tested on not replacement case yet)
                if (not replacement):
                    pool_inds_split = list(set(pool_inds_split) - set(query_ann_inds_split))
                    Xpool_split = eval('X_'+dataset)[pool_inds_split]


                PIs, w_split_mus_prev_steps = split_mfcs_design.compute_confidence_sets_design(Xtrain_split, Xcal_split, ytrain_split, ycal_split, Xtest_split, ytest_split, Xpool_split, w_split_mus_prev_steps, method_names, weight_bounds, t_cal = t_cal, X_dataset = eval('X_'+dataset), weight_depth_maxes=weight_depth_maxes, lmbda = lmbda, bandwidth = 1.0, alpha=alpha, n_initial_all = n_initial_all, n_dataset = eval('n_'+dataset), replacement = replacement) #, add_to_cal = add_to_cal
                

                ## Add point that was queried for annotation to the training or calibration data 
                U = np.random.uniform()
                if (U <= add_to_train_split_prob):
                    ## Add to training data

                    for q_ann in query_ann_inds_split:
                        train_inds_split.append(q_ann) ## Add queried samples to training set

                    Xtrain_split = eval('X_'+dataset)[train_inds_split]
                    ytrain_split = np.concatenate((ytrain_split, ytest_split))

                    ## If added point to training data, then do not need to keep track of this muh function
                    ## Only the muh functions used for querying points eventually added to calibration
                    ## will be needed later.
                    w_split_mus_prev_steps = w_split_mus_prev_steps[:-1]

                else:
                    ## Add to calibration data

                    for q_ann in query_ann_inds_split:
                        cal_inds_split.append(q_ann) ## Add queried samples to training set

                    Xcal_split = eval('X_'+dataset)[cal_inds_split]
                    ycal_split = np.concatenate((ycal_split, ytest_split))

                    ## If adding to calibration, incrememnt the number of actively selected calibration points
                    t_cal += 1

                MSE_split = np.mean((y_preds_val_split - yval)**2)


                for method in method_names:

                    coverage_by_seed = ((PIs[method]['lower'] <= ytest_split)&(PIs[method]['upper'] >= ytest_split)).mean()
                    muh_test_by_seed = ytest_preds_split.mean()
                    coverage_all = ((PIs[method]['lower'] <= ytest_split)&(PIs[method]['upper'] >= ytest_split))
                    muh_test_all = ytest_preds_split
                    ytest_method = ytest_split
                    MSE = MSE_split

                    width_by_seed = (PIs[method]['upper'] - PIs[method]['lower']).median()
                    width_all = (PIs[method]['upper'] - PIs[method]['lower'])

                    results_by_seed.loc[len(results_by_seed)]=\
                            [seed,step, dataset, muh, method,coverage_by_seed,width_by_seed,MSE]

                    for test_pt in range(0, n_test):
                        results_all.loc[len(results_all) + test_pt]=[seed,step, test_pt,dataset,muh,method,\
                                                                   coverage_all[test_pt],width_all[test_pt],\
                                                                     muh_test_all[test_pt], ytest_method[test_pt]]
            if (((seed+1) % 100) == 0):
                results_by_seed.to_csv(os.getcwd().removesuffix('bash_scripts') + 'results/'+ str(date.today()) + '_SplitDesignExpts_' + dataset + '_' + muh + '_nInitial' + str(n_initial_all) + '_steps' + str(n_steps) + '_nseed' + str(n_seed) + '_lmbda' + str(lmbda) + '_wDepths' + str_dep_maxes + '_propTrain'+ str(p_split_train) +'_addTrainProb'+str(add_to_train_split_prob)+'_noise'+str(noise_magnitude)+'_probBoundInf' + str(prob_bound_inf) +'_BySeed.csv',index=False)

    end_time = time.time()

    print("Total time (minutes) : ", (end_time - start_time)/60)
    
    print(os.getcwd().removesuffix('bash_scripts') + 'results/'+ str(date.today()) + '_SplitDesignExpts_' + dataset + '_' + muh + '_nInitial' + str(n_initial_all) + '_steps' + str(n_steps) + '_nseed' + str(n_seed) + '_lmbda' + str(lmbda) + '_wDepths' + str_dep_maxes + '_propTraini'+ str(p_split_train) +'_addTrainProb'+str(add_to_train_split_prob)+'_noise'+str(noise_magnitude) + '_probBoundInf' + str(prob_bound_inf))

    results_by_seed.to_csv(os.getcwd().removesuffix('bash_scripts') + 'results/'+ str(date.today()) + '_SplitDesignExpts_' + dataset + '_' + muh + '_nInitial' + str(n_initial_all) + '_steps' + str(n_steps) + '_nseed' + str(seed_initial) + '_lmbda' + str(lmbda) + '_wDepths' + str_dep_maxes + '_propTrain'+ str(p_split_train) +'_addTrainProb'+str(add_to_train_split_prob)+'_noise'+str(noise_magnitude)+'_probBoundInf' + str(prob_bound_inf) +'_BySeed.csv',index=False)

    results_all.to_csv(os.getcwd().removesuffix('bash_scripts') + 'results/'+ str(date.today()) + '_SplitDesignExpts_' + dataset + '_' + muh + '_nInitial' + str(n_initial_all) + '_steps' + str(n_steps) + '_nseed' + str(seed_initial) + '_lmbda' + str(lmbda) + '_wDepths' + str_dep_maxes + '_propTrain'+ str(p_split_train) +'_addTrainProb'+str(add_to_train_split_prob)+'_noise'+str(noise_magnitude)+'_probBoundInf' + str(prob_bound_inf) +'_ALL.csv',index=False)

