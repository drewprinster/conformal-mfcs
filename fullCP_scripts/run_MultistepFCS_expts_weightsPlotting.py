import os
import sys
import time
from importlib import reload
module_path = os.path.abspath(os.path.join('../')) ## Drew modified this from '../' to './'
if module_path not in sys.path:
    sys.path.append(module_path)
    
import numpy as np
    
import assay_20240123 as assay
import calibrate_20240219_FIXED_weightsPlotting as cal
import datetime
from datetime import date

import pandas as pd
from argparse import ArgumentParser
from datetime import date
import tqdm


## python run_MultistepFCSexpts.py --fitness_str red --n_train_initial 15 --lmbdas 4 --n_seed 10 --reg 5 --depth_max 2

if __name__ == "__main__":
    
    parser = ArgumentParser(description='Run multistep FCS experiments.')
    
    parser.add_argument('--fitness_str', type=str, default='red', help='Red or blue fluorescence experiments.')
    parser.add_argument('--n_train_initial', type=int, help='Initial number of training points', required = True)
    parser.add_argument('--lmbdas', nargs='+', help='Values of lmbda to try', required = True)
    parser.add_argument('--n_seed', type=int, default=500, help='Number of trials')
    parser.add_argument('--n_steps', type=int, default=4, help='Number of MFCS steps (active learning iterations)')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha value corresponding to 1-alpha target coverage')
    parser.add_argument('--reg', type=float, default=1, help='regularization strength for ridge regression')
#     parser.add_argument('--K_vals', nargs='+', help='Values of K to try', required = True)
    parser.add_argument('--muh', type=str, default='ridge', help='Muh predictor.')
    parser.add_argument('--depth_max', type=int, default=2, help='Maximum recursion depth for weight adjustment.')
    parser.add_argument('--seed_initial', type=int, default=0, help='Initial seed to start at..')

    
    args = parser.parse_args()
    fitness_str = dataset = args.fitness_str
#     n_trains = [int(n_train) for n_train in args.n_trains]
    n_train_initial = args.n_train_initial
    lmbdas = [int(lmbda) for lmbda in args.lmbdas]
    n_seed = args.n_seed
#     n_seed = 4 ## For now override to 4 trials
    alpha = args.alpha
#     K_vals = [int(K) for K in args.K_vals]
#     K_vals = [2, 4, 6] ## For now override to K=2,4
    muh = args.muh
    reg = args.reg
    weight_adj_depth_max = args.depth_max
    n_steps = args.n_steps
    seed_initial = args.seed_initial

    reload(cal)
    reload(assay)

#     alpha = 0.1                           # miscoverage
    # n_trains = [20] #96, 192, 384             # number of training points
    ntrain2reg = {n_train_initial: reg} # ridge regularization strength (gamma in code and paper), 192: 1, 384: 1
#     n_seed = 200                      # number of random trials
#     lmbdas = [2]                 # lambda, inverse temperature [0, 2, 4, 6]
    y_increment = 0.02                    # grid spacing between candidate label values, \Delta in paper
    ys = np.arange(0, 2.21, y_increment)  # candidate label values, \mathcal{Y} in paper
    order = 2                             # complexity of features. 1 encodes the AA at each site,
                                          # 2 the AAs at each pair of sites,
                                          # 3 the AAs at each set of 3 sites, etc.

    print('Running FullCPMultistepDesignExpts_' + dataset + '_' + muh + '_ntrain_init' + str(n_train_initial) + '_steps' + str(n_steps) + '_nseed' + str(n_seed) + '_lmbda' + str(lmbdas[0]) + '_reg' + str(reg) + '_depth' + str(weight_adj_depth_max))
    

    # results_by_seed = pd.DataFrame(columns = ['seed', 'step', 'dataset', 'muh_fun','method','coverage','width', 'MSE'])
    results_all = pd.DataFrame(columns = ['seed','step', 'dataset','muh_fun','method','coverage','width', 'muh_test', 'y_test'])
    method_names = ['full_ex', 'full_1fcs', 'full_mfcs']
    cover_curr = np.zeros(len(method_names))
    width_curr = np.zeros(len(method_names))


    # likelihood under training input distribution, p_X in paper (uniform distribution)
    ptrain_fn = lambda x: (1.0 / np.power(2, 13)) * np.ones([x.shape[0]])
    # for fitness_str in ['red', 'blue']:

    # featurize all sequences in combinatorially complete dataset
    data = assay.PoelwijkData(fitness_str, order=order)


    reg = ntrain2reg[n_train_initial]
    fcs1 = cal.ConformalRidgeMultistepFeedbackCovariateShift(ptrain_fn, ys, data.X_nxp, reg)
    mfcs = cal.ConformalRidgeMultistepFeedbackCovariateShift(ptrain_fn, ys, data.X_nxp, reg)
    scs = cal.ConformalRidgeStandardCovariateShift(ptrain_fn, ys, data.X_nxp, reg)
    ex = cal.ConformalRidgeExchangeable(ptrain_fn, ys, data.X_nxp, reg)

    timestamp = time.time()
    value = datetime.datetime.fromtimestamp(timestamp)
    print("start time : ", value.strftime('%Y-%m-%d %H:%M:%S'))


    for l, lmbda in enumerate(lmbdas):


        for seed in range(seed_initial,seed_initial+ n_seed):
            print("seed = ", seed)
            n_train = n_train_initial

            for step in range(1,n_steps+1):
                print("   step = ", step)
                if (step == 1):
                    # At first step, sample training data uniformly and get designed data
                    Xtrain_nxp, ytrain_n, Xtest_1xp, ytest_1, pred_1 = assay.get_training_and_designed_data(
                        data, n_train, reg, lmbda, seed=seed)



                else:
                    # At subsequent steps, add previous test point to training and get new designed data

                    Xtrain_nxp = np.vstack([Xtrain_nxp, Xtest_1xp])
                    ytrain_n = np.concatenate((ytrain_n, ytest_1))
                    n_train += 1

                    Xtest_1xp, ytest_1, pred_1 = assay.sample_new_designed_data(data, Xtrain_nxp, ytrain_n, n_train, reg, lmbda, seed=seed)


                # construct confidence set under feedback covariate shift
                weight_adj_depth = min(weight_adj_depth_max, step)
                exset, _  = ex.get_confidence_set(Xtrain_nxp, ytrain_n, Xtest_1xp, lmbda, alpha=alpha)
                fset1, _ = fcs1.get_confidence_set(Xtrain_nxp, ytrain_n, Xtest_1xp, lmbda, alpha=alpha, n_step_curr=1)
                mfset, _, w_n1xy = mfcs.get_confidence_set(Xtrain_nxp, ytrain_n, Xtest_1xp, lmbda, alpha=alpha, n_step_curr=weight_adj_depth)


                ## Record coverages and widths
                cover_curr[0] = cal.is_covered(ytest_1[0], exset, y_increment)
                cover_curr[1] = cal.is_covered(ytest_1[0], fset1, y_increment)
                cover_curr[2] = cal.is_covered(ytest_1[0], mfset, y_increment)

                width_curr[0] = exset.size * y_increment
                width_curr[1] = fset1.size * y_increment
                width_curr[2] = mfset.size * y_increment



                ## Add results for each method
                for i, method in enumerate(method_names):
                    results_all.loc[len(results_all)]=[seed,step, dataset,muh,method,\
                                                                cover_curr[i],width_curr[i],\
                                                                pred_1[0], ytest_1[0]]


            if (((seed+1) % 50) == 0):
                results_all.to_csv(os.getcwd().removesuffix('bash_scripts') + '/results/'+ str(date.today()) + '_FullCPMultistepDesignExpts_' + dataset + '_' + muh + '_ntrain_init' + str(n_train_initial) + '_steps' + str(n_steps) + '_nseed' + str(n_seed) + '_lmbda' + str(lmbda) + '_reg' + str(reg) + '_depth' + str(weight_adj_depth_max) + '_seedinit' + str(seed_initial) + '_PIs_replaceTrue_v2.csv',index=False)



    results_all.to_csv(os.getcwd().removesuffix('bash_scripts') + '/results/'+ str(date.today()) + '_FullCPMultistepDesignExpts_' + dataset + '_' + muh + '_ntrain_init' + str(n_train_initial) + '_steps' + str(n_steps) + '_nseed' + str(n_seed) + '_lmbda' + str(lmbda) + '_reg' + str(reg) + '_depth' + str(weight_adj_depth_max) + '_seedinit' + str(seed_initial) + '_PIs_replaceTrue_v2.csv',index=False)

    timestamp = time.time()
    value = datetime.datetime.fromtimestamp(timestamp)
    print("end time : ", value.strftime('%Y-%m-%d %H:%M:%S'))


