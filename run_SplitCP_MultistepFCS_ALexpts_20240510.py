import os
import sys
import time
from importlib import reload
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import numpy as np
    
import assay
import calibrate_ALexpts_20240510 as cal

## Drew added
import tqdm
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
    

## Drew added
def MINSTD_pseudo_rng(x):
    (16807 * x) % 2147483647

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
            # print("iterated through ", i , " out of ", len(exp_vars_pool_sorted_descending))
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
    
    b = exp_vars_pool_sorted_descending[mid]
    
    total_weight_cal = np.sum(np.minimum(exp_vars_cal, b))
    unnormalized_weights_pool = np.minimum(exp_vars_pool_sorted_descending, b)
    inds_infty_pool = np.where(unnormalized_weights_pool / (total_weight_cal + unnormalized_weights_pool) >= alpha)[0]

        
    if (start + 1 == end):
        # print("len inds_infty_pool : ", len(inds_infty_pool))
        return start, exp_vars_pool_sorted_descending[start]
    
    
    # print("start : ", start, "mid : ", mid, "end : ", end)

    
    
                
    if (np.sum(unnormalized_weights_pool[inds_infty_pool]) / np.sum(unnormalized_weights_pool) <= beta):
        # print("IF case")

        return compute_weight_bound_binary_search(beta, exp_vars_cal, exp_vars_pool_sorted_descending, start=mid, end=end, alpha=0.1)

    else:
        # print("ELSE case")
        return compute_weight_bound_binary_search(beta, exp_vars_cal, exp_vars_pool_sorted_descending, start=start, end=mid, alpha=0.1)
            
    



## test
# if __name__ == "__main__":

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run multistep FCS experiments.')
    
    parser.add_argument('--dataset', type=str, default='airfoil', help='Dataset name.')
    parser.add_argument('--n_train_initial', type=int, help='Initial number of training points', required = True)
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
    parser.add_argument('--weight_adj_depth_maxes', nargs='+', default=[1,2,3,4], help='Depths of weight adjustments to include')
    parser.add_argument('--kernel_str', type=str, default='dot_white', help='Kernel type for GP')
    parser.add_argument('--noise_magnitude', type=float, default=0.1, help='Positive value controlling noise magnitude. 1 means noise is one std')
    parser.add_argument('--initial_sampling_bias', type=float, default=1.0, help='Expontential tilting of how training & calibration data are sampled from the full dataset, w.r.t. first PCA of the data.')
    parser.add_argument('--save_weight_computations', type=bool, default=False, help='Bool about whether to save weights')
    parser.add_argument('--noise_level', type=float, default=1.0, help='White kernel noise level')
    parser.add_argument('--sigma_0', type=float, default=1.0, help='Dot product kernel noise level')
    parser.add_argument('--aci_step_size', type=float, default=0.005, help='ACI step size gamma')
    parser.add_argument('--beta', type=float, default=0.0, help='Beta : acceptable probability of infinite interval width')

    ## python run_SplitCP_MultistepFCS_expts.py --dataset airfoil --n_train_initial 48 --lmbdas 10 --n_seed 200 --p_split_train 0.6 
    
    args = parser.parse_args()
    dataset = args.dataset
#     n_trains = [int(n_train) for n_train in args.n_trains]
    n_train_initial = args.n_train_initial
    lmbdas = [float(lmbda) for lmbda in args.lmbdas]
    lmbda = lmbdas[0]
    tilt_factor = lmbda
    n_seed = args.n_seed
#     n_seed = 4 ## For now override to 4 trials
    alpha = args.alpha
#     K_vals = [int(K) for K in args.K_vals]
    K_vals = [2, 4, 6] ## For now override to K=2,4
    muh = args.muh
    n_steps = args.n_steps
    n_queries_ann = args.n_queries_ann
    n_test = n_queries_ann
    replacement = args.replacement
    add_to_train_split_prob = args.add_to_train_split_prob
    weight_adj_depth_maxes = [int(dep) for dep in args.weight_adj_depth_maxes]
    seed_initial = args.seed_initial
    n_val = args.n_val
    p_split_train = args.p_split_train
    str_dep_maxes = ''
    for d in weight_adj_depth_maxes:
        str_dep_maxes += str(d)
    kernel_str = args.kernel_str
    noise_magnitude = args.noise_magnitude
    initial_sampling_bias = args.initial_sampling_bias
    save_weight_computations = args.save_weight_computations
    noise_level = args.noise_level
    sigma_0 = args.sigma_0
    aci_step_size = args.aci_step_size
    beta = args.beta
    
    
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
    for d in weight_adj_depth_maxes:
        method_names.append('wsplit_' + str(d))
        
    method_names.append('aci')
    
    print("Running with methods : ", method_names)
    
    
    
    print('Running with '+ dataset + '_' + muh + '_itrain' + str(n_train_initial) + '_steps' + str(n_steps) + '_nseed' + str(n_seed) + '_iseed' + str(seed_initial) + '_tilt' + str(lmbda) + '_wAdjs' + str_dep_maxes + '_PIs_addTrainProb'+str(add_to_train_split_prob)+'_replace'+str(replacement)+'_kernel'+kernel_str +'pcaExpSampling'+str(initial_sampling_bias) + str(aci_step_size)+ '_beta' + str(beta))
    


    n_train_proper_initial = int(np.floor(n_train_initial * p_split_train))
    n_cal_initial = n_train_initial - n_train_proper_initial
    
    
    results_by_seed = pd.DataFrame(columns = np.concatenate([['seed', 'step', 'n_cal_active', 'dataset', 'muh_fun','method','coverage','width', 'MSE', 'muh_test', 'y_test', 'B', 'prop_B', 'alpha_aci'], ['w_cal' + str(i) for i in range(0, n_cal_initial+n_steps-1)], ['w_test']]))

    

    #     # likelihood under training input distribution, p_X in paper (uniform distribution)
    ptrain_fn = cal.KDE_density_estimates

    


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

    elif (dataset == 'wave'):
        wave = pd.read_csv(os.getcwd().removesuffix('bash_scripts') + '/datasets/WECs_DataSet/Adelaide_Data.csv', header = None)
        X_wave = wave.iloc[0:4000, 0:48].values
        Y_wave = wave.iloc[0:4000, 48].values
        n_wave = len(Y_wave)
        errs_wave = compute_errors_for_measurements(X_wave, Y_wave)
        print("X_wave shape : ", X_wave.shape)

    elif (dataset == 'superconduct'):
        superconduct = pd.read_csv(os.getcwd().removesuffix('bash_scripts') + '/datasets/superconduct/train.csv')
        X_superconduct = superconduct.iloc[0:2000, 0:81].values
        Y_superconduct = superconduct.iloc[0:2000, 81].values
        n_superconduct = len(Y_superconduct)
        errs_superconduct = compute_errors_for_measurements(X_superconduct, Y_superconduct)
        print("X_superconduct shape : ", X_superconduct.shape)

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
        meps_data = np.loadtxt(os.getcwd().removesuffix('bash_scripts') + '/datasets/meps_data.txt')
        X_meps = meps_data[:,:-1]
        Y_meps = meps_data[:,-1]
        n_meps = len(Y_meps)
        errs_meps = compute_errors_for_measurements(X_meps, Y_meps)
        print("X_meps shape : ", X_meps.shape)

    elif (dataset == 'blog'):
        blog_data = np.loadtxt(os.getcwd().removesuffix('bash_scripts') + '/datasets/blogData_train.csv',delimiter=',')
        X_blog = blog_data[:,:-1]
        Y_blog = np.log(1+blog_data[:,-1])
        n_blog = len(Y_blog)
        errs_blog = compute_errors_for_measurements(X_blog, Y_blog)
    

    elif (dataset == 'red_protein'):

        #######################
        #### This section added for protein design 
        #######################
        os.chdir('/home/drewprinster/conformal-mfcs/fluorescence/')
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


        os.chdir('/home/drewprinster/conformal-mfcs/')
        #######################

    elif (dataset == 'blue_protein'):

        #######################
        #### This section added for protein design 
        #######################
        os.chdir('/home/drewprinster/conformal-mfcs/')
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


        os.chdir('/home/drewprinster/conformal-mfcs/')
        #######################


    X_all = eval('X_'+dataset)
    all_inds = np.arange(eval('n_'+dataset))

    jaw_fcs_active = cal.JAWFeedbackCovariateShiftActive(muh_fun, ptrain_fn, X_all)



    for seed in range(seed_initial, seed_initial + n_seed):
        # print("seed : ", seed)
        Y_all = noisy_measurements(all_inds, eval('Y_'+dataset), eval('errs_'+dataset), noise_magnitude, seed)

        ## Note: Validation set won't change, train and pool will
        np.random.seed(seed)
        val_inds = list(np.random.choice(eval('n_'+dataset),n_val,replace=False))
        non_val_inds = np.setdiff1d(all_inds, val_inds)
        X_nonval = X_all[non_val_inds]

        ## Added these three lines to draw data form a random "orthant" of the data, based on PCA
        X_nonval_pca = get_PCA(X_nonval).flatten()
        X_nonval_pca_minmax = (X_nonval_pca - min(X_nonval_pca)) / (max(X_nonval_pca) - min(X_nonval_pca))
        X_nonval_pca_minmax_nonvals_exp = np.exp(X_nonval_pca_minmax * initial_sampling_bias)
        X_nonval_pca_minmax_nonvals_exp_normed = X_nonval_pca_minmax_nonvals_exp / np.sum(X_nonval_pca_minmax_nonvals_exp)
        
        train_inds = list(np.random.choice(non_val_inds, n_train_initial, replace=replacement, p=X_nonval_pca_minmax_nonvals_exp_normed))
    
        ## Pool inds are those not in training or validation data.
        pool_inds = list(np.setdiff1d(np.setdiff1d(np.arange(eval('n_'+dataset)),train_inds), val_inds)) 

        ## Initialize train and pool data for no sample splitting
        Xtrain = eval('X_'+dataset)[train_inds]
        ytrain = Y_all[train_inds]
        Xpool = eval('X_'+dataset)[pool_inds]
        ypool = Y_all[pool_inds]


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
        # w_LOO_mus_prev_steps = []
        exp_vals_pool_list_of_vecs_all_steps = []
        w_split_mus_prev_steps = []

        ## Keep track of number of points currently in the calibration set that were actively selected
        num_cal_actively_selected = 0
        
        ## Initialize alpha_aci_curr to alpha
        alpha_aci_curr = alpha
        
        weight_bounds = []

        for step in range(n_steps):
            # print("step : ", step)

            Xtest = None
            ytest = None
            y_preds_val = None


            ####### ******* Sample splitting ********* ########

            ## 1. Fit Gaussian process regression and use it to select queries from pool
            gpr_split = muh_fun.fit(Xtrain_split, ytrain_split)
            

            ############ minmax normed
            ## 2. Compute unnormalized weights for pool
            # print("len(Xpool_split) ", len(Xpool_split))
            y_preds_pool_split, std_preds_pool_split_ = gpr_split.predict(Xpool_split, return_std=True) ## 20240117: Changed from Xpool_split to X_all
            std_preds_pool_split = get_f_std(std_preds_pool_split_, gpr_split)

            var_preds_pool_split = std_preds_pool_split**2
            var_preds_pool_split_minmax_normed = (var_preds_pool_split) / (max(var_preds_pool_split) - min(var_preds_pool_split))
            exp_var_preds_pool_split = np.exp(var_preds_pool_split_minmax_normed * tilt_factor)
            
            exp_vals_pool_list_of_vecs_all_steps.append(exp_var_preds_pool_split) ## Added 20240326 : Saving exp pool vals for later
            
            
            exp_var_preds_pool_split_sorted = np.sort(exp_var_preds_pool_split)
        
        
        
            ## 3. Compute unnormalized weights for calibration data
            ycal_preds_split, std_preds_cal_split_ = gpr_split.predict(Xcal_split, return_std=True)
            std_preds_cal_split = get_f_std(std_preds_cal_split_, gpr_split)

            var_preds_cal_split = std_preds_cal_split**2
            var_preds_cal_split_minmax_normed = (var_preds_cal_split) / (max(var_preds_pool_split) - min(var_preds_pool_split))
            exp_var_preds_cal_split = np.exp(var_preds_cal_split_minmax_normed * tilt_factor)
            
            
            ## 4. Compute bound on weight
            i_B, B = compute_weight_bound_binary_search(beta, exp_var_preds_cal_split, exp_var_preds_pool_split_sorted, start=0, end=len(exp_var_preds_pool_split_sorted)-1, alpha=alpha)
            weight_bounds.append(B)
            
            
            
            ## 5. Compute normalized pool weights and use those for sampling a query point
            probs_pool_split = np.minimum(exp_var_preds_pool_split, B) / np.sum(np.minimum(exp_var_preds_pool_split, B))
            
            
            query_ann_inds_split = list(np.random.choice(pool_inds_split, n_queries_ann, replace=replacement, p=probs_pool_split)) ## 20240117: Changed to pool_inds_split to all_inds; replace=True

            Xtest_split = eval('X_'+dataset)[query_ann_inds_split]

            ytest_split = noisy_measurements(query_ann_inds_split, eval('Y_'+dataset), eval('errs_'+dataset), noise_magnitude, seed)

            ## 3. For later MSE calculation: Compute predictions on validation data
            y_preds_val_split, std_preds_val_split_ = gpr_split.predict(Xval, return_std=True)
            std_preds_val_split = get_f_std(std_preds_val_split_, gpr_split)

            ytest_preds_split = gpr_split.predict(Xtest_split)

            ### Drew added this 20240111

            if (not replacement):

                pool_inds_split = list(set(pool_inds_split) - set(query_ann_inds_split))
                Xpool_split = eval('X_'+dataset)[pool_inds_split]
                


            PIs, w_split_mus_prev_steps, weights_normalized_wsplit_all = jaw_fcs_active.compute_PIs_active(Xtrain, ytrain, Xtest, ytest, Xtrain_split, Xcal_split, ytrain_split, ycal_split, Xtest_split, ytest_split, Xpool_split, w_split_mus_prev_steps, exp_vals_pool_list_of_vecs_all_steps, method_names, n_step_curr=num_cal_actively_selected+1, X_dataset = eval('X_'+dataset), n_cal_initial = n_cal_initial, alpha_aci_curr = alpha_aci_curr, weight_bounds = weight_bounds, weight_adj_depth_maxes=weight_adj_depth_maxes, tilt_factor = tilt_factor, bandwidth = 1.0, beta=beta, alpha=alpha, K_vals = K_vals, n_train_initial = n_train_initial, n_dataset = eval('n_'+dataset), replacement = replacement)

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
                
                # print("num_cal_actively_selected ", num_cal_actively_selected)
                
                if (m_i == 0 or method == 'aci'):
                    cal_test_weights = np.concatenate([np.repeat(1 / (n_cal_initial + num_cal_actively_selected + 1), \
                                                                 n_cal_initial + num_cal_actively_selected), \
                                                       np.repeat(0, n_steps-num_cal_actively_selected-1), \
                                                       [1/(n_cal_initial + num_cal_actively_selected + 1)]])
                    
                    
                else:
                    cal_test_weights = np.concatenate([weights_normalized_wsplit_all[m_i-1][:-1], np.repeat(0, n_steps-num_cal_actively_selected-1), [weights_normalized_wsplit_all[m_i-1][-1]]])
                    
                
                prop_B = i_B / len(pool_inds_split)
                
                results_by_seed.loc[len(results_by_seed)]=\
                        np.concatenate([[seed,step,num_cal_actively_selected, dataset, muh, method,coverage_by_seed,width_by_seed,MSE, muh_test_all[0], ytest_method[0], B, prop_B, alpha_aci_curr], cal_test_weights])

                if (method == 'aci'):
                    
                    if (PIs[method]['lower'].values[0] <= ytest_split[0] and PIs[method]['upper'].values[0] >= ytest_split[0]):
                        alpha_aci_curr = min(alpha_aci_curr + aci_step_size * (alpha),1)
                    else:
                        alpha_aci_curr = max(alpha_aci_curr + aci_step_size * (alpha - 1),0)
                    
                    
                    
                    
            ## ****** Sample Splitting: Prepare for next active learning iteration ******
            ## Add point that was queried for annotation to the training data 
            ## & remove queried point from pooled data
            U = np.random.uniform()
            if (U <= add_to_train_split_prob):
    #             print("add to train")
                ### Update train data for sample splitting
                ### Note: Calibration set for split won't change

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
                ### Note: Training set for split won't change

                for q_ann in query_ann_inds_split:
                    cal_inds_split.append(q_ann) ## Add queried samples to training set

                Xcal_split = eval('X_'+dataset)[cal_inds_split]
                ycal_split = np.concatenate((ycal_split, ytest_split))


                ## Incrememnt the number of actively selected calibration points
                num_cal_actively_selected += 1

                
                
                    
        if (((seed+1) % 50) == 0):
            results_by_seed.to_csv(os.getcwd().removesuffix('bash_scripts') + '/results/'+ str(date.today()) + '_ALExpts_v7_' + dataset + '_' + muh + '_itrain' + str(n_train_initial) + '_steps' + str(n_steps) + '_nseed' + str(n_seed) + '_iseed' + str(seed_initial) + '_tilt' + str(tilt_factor) + '_wAdjs' + str_dep_maxes + '_PIs_propTraini'+ str(p_split_train) +'_addTrainProb'+str(add_to_train_split_prob)+'_replace'+str(replacement)+'_noise'+str(noise_magnitude)+'_BySeed_pcaExpSampling'+str(initial_sampling_bias)+ '_GPRnoise' + str(noise_level)  + '_sigDotProd' + str(sigma_0)+ '_aciStepSize' + str(aci_step_size)+ '_beta' + str(beta) + '_funcB.csv',index=False)

    end_time = time.time()

    print("Total time (minutes) : ", (end_time - start_time)/60)

    results_by_seed.to_csv(os.getcwd().removesuffix('bash_scripts') + '/results/'+ str(date.today()) + '_ALExpts_v7_' + dataset + '_' + muh + '_itrain' + str(n_train_initial) + '_steps' + str(n_steps) + '_nseed' + str(n_seed) + '_iseed' + str(seed_initial) + '_tilt' + str(tilt_factor) + '_wAdjs' + str_dep_maxes + '_PIs_propTraini'+ str(p_split_train) +'_addTrainProb'+str(add_to_train_split_prob)+'_replace'+str(replacement)+'_noise'+str(noise_magnitude)+'_BySeed_pcaExpSampling'+str(initial_sampling_bias)+ '_GPRnoise' + str(noise_level) + '_sigDotProd' + str(sigma_0)+ '_aciStepSize' + str(aci_step_size)+'_beta' + str(beta) +'_funcB.csv',index=False)
