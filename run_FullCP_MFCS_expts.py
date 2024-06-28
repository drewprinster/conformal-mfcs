import os
import sys
import time
from importlib import reload
    
import numpy as np
    
import assay_mfcs as assay
import calibrate_DesignExperiments as cal
import datetime
from datetime import date

import pandas as pd
from argparse import ArgumentParser
from datetime import date


## Example useage
## python run_FullCP_MFCS_expts.py --fitness_str blue --n_train_initial 32 --lmbdas 8 --n_seed 20 --n_steps 5 --reg 0.01 --depth_max 2

if __name__ == "__main__":
    
    parser = ArgumentParser(description='Run Full CP multistep FCS experiments.')
    
    parser.add_argument('--fitness_str', type=str, default='red', help="Color for fluorescent protein experiments: 'red' or 'blue'")
    parser.add_argument('--n_train_initial', type=int, help='Initial number of training points (= # cal for full CP).', required = True)
    parser.add_argument('--lmbdas', nargs='+', help='Values of lmbda (shift magnitude) to try.', required = True)
    parser.add_argument('--n_seed', type=int, default=500, help='Number of trials.')
    parser.add_argument('--n_steps', type=int, default=4, help='Number of MFCS steps (active learning iterations).')
    parser.add_argument('--alpha', type=float, default=0.1, help='Target miscoverage (i.e., 1-alpha is target coverage).')
    parser.add_argument('--reg', type=float, default=1, help='Regularization strength for ridge regression.')
    parser.add_argument('--muh', type=str, default='ridge', help='String to specify muh predictor.')
    parser.add_argument('--depth_max', type=int, default=2, help="Maximum estimation depth $d$ for MFCS weight computation, Eq. (9) in paper.")
    parser.add_argument('--seed_initial', type=int, default=0, help='Initial seed to start at.')

    
    args = parser.parse_args()
    fitness_str = dataset = args.fitness_str
    n_train_initial = args.n_train_initial
    lmbdas = [float(lmbda) for lmbda in args.lmbdas]
    n_seed = args.n_seed
    alpha = args.alpha
    muh = args.muh
    reg = args.reg
    depth_max = args.depth_max
    n_steps = args.n_steps
    seed_initial = args.seed_initial

    reload(cal)
    reload(assay)
    
    method_names = ['full_ex', 'full_1fcs', 'full_mfcs']

    
    print('Running FullCPMultistepDesignExpts_' + dataset + '_' + muh + '_ntrain_init' + str(n_train_initial) + '_steps' + str(n_steps) + '_nseed' + str(n_seed) + '_lmbda' + str(lmbdas[0]) + '_reg' + str(reg) + '_depth' + str(depth_max))
    

    ## Arrays for temporary results and data frame for all results
    cover_curr = np.zeros(len(method_names))
    width_curr = np.zeros(len(method_names))
    results_all = pd.DataFrame(columns = ['seed','step', 'dataset','muh_fun','method','coverage','width', 'muh_test', 'y_test'])
    

    ## Likelihood under training input distribution, p_X in paper (uniform distribution in protein design experiments).
    ## Due to initializing with IID *uniform random* samples, the following two lines will yield equivalent results, but both
    ## are provided to correspond to different factorizations given in Appendix B.1 (direct likelihood factorization approach)
    ## and Appendix B.3 (likelihood-*ratio*-based factorization) respectively.
    ptrain_fn = lambda x: np.ones([x.shape[0]]) ## Trivial denominator (=1) for direct likelihood factorization (as in Appendix B.1 in paper)
    # ptrain_fn = lambda x: (1.0 / np.power(2, 13)) * np.ones([x.shape[0]]) ## Likelihood-ratio based factorization (see Appendix B.3 in paper)

    
    y_increment = 0.02                    # grid spacing between candidate label values
    ys = np.arange(0, 2.21, y_increment)  # candidate label values, \mathcal{Y} in paper
    order = 2                             # complexity of features. 1 encodes the AA at each site,
                                          # 2 the AAs at each pair of sites,
                                          # 3 the AAs at each set of 3 sites, etc.
    data = assay.PoelwijkData(fitness_str, order=order) # featurize all sequences in combinatorially complete dataset

    
    ## Load classes for Full CP baselines and MFCS Full CP (proposed) method)
    ex = cal.FullConformalRidgeExchangeable(ptrain_fn, ys, data.X_nxp, reg) ## Class for Exchangeable Full CP
    fcs1 = cal.FullConformalRidgeMultistepFeedbackCovariateShift(ptrain_fn, ys, data.X_nxp, reg) ## Class for One-Step FCS Full CP
    mfcs = cal.FullConformalRidgeMultistepFeedbackCovariateShift(ptrain_fn, ys, data.X_nxp, reg) ## Class for MFCS Full CP (proposed)
    

    timestamp = time.time()
    value = datetime.datetime.fromtimestamp(timestamp)
    print("start time : ", value.strftime('%Y-%m-%d %H:%M:%S'))

    
    ## Loop for repeating experiment with different lmbdas (shift magnitudes)
    for l, lmbda in enumerate(lmbdas):

        
        ## Loop for repeating experiment with different random seeds (which changes the training sample)
        for seed in range(seed_initial,seed_initial+ n_seed):
            print("seed = ", seed)
            n_train = n_train_initial

            
            ## Loop for each step in the multistep design process (in each step >= 2, one point is queried by the ML model, 
            ## a prediction set is computed for the point, and it is then subsequently labeled and added to training).
            for step in range(1,n_steps+1):
                print("   step = ", step)
                if (step == 1):
                    # At first step, sample training data uniformly and query designed (test) data
                    Xtrain_nxp, ytrain_n, Xtest_1xp, ytest_1, pred_1 = assay.get_training_and_designed_data(
                        data, n_train, reg, lmbda, seed=seed)


                else:
                    # At subsequent steps, add previous test point to training and query new designed (test) data
                    Xtrain_nxp = np.vstack([Xtrain_nxp, Xtest_1xp])
                    ytrain_n = np.concatenate((ytrain_n, ytest_1))
                    n_train += 1

                    Xtest_1xp, ytest_1, pred_1 = assay.sample_new_designed_data(data, Xtrain_nxp, ytrain_n, n_train, reg, lmbda, seed=seed)
                    

                # Construct confidence sets for each Full CP method
                depth = min(depth_max, step)
                exset, _ = ex.compute_confidence_sets(Xtrain_nxp, ytrain_n, Xtest_1xp, lmbda, alpha=alpha)
                fset1, _ = fcs1.compute_confidence_sets(Xtrain_nxp, ytrain_n, Xtest_1xp, lmbda, alpha=alpha, depth_max=1)
                mfset, _ = mfcs.compute_confidence_sets(Xtrain_nxp, ytrain_n, Xtest_1xp, lmbda, alpha=alpha, depth_max=depth)

                ## Record coverages and widths
                cover_curr[0] = cal.is_covered(ytest_1[0], exset, y_increment)
                cover_curr[1] = cal.is_covered(ytest_1[0], fset1, y_increment)
                cover_curr[2] = cal.is_covered(ytest_1[0], mfset, y_increment)

                width_curr[0] = exset.size * y_increment
                width_curr[1] = fset1.size * y_increment
                width_curr[2] = mfset.size * y_increment


                ## Add results (for current seed) to dataframe for each method
                for i, method in enumerate(method_names):
                    results_all.loc[len(results_all)]=[seed,step, dataset,muh,method,cover_curr[i],width_curr[i],pred_1[0], ytest_1[0]]

              
            ## Save interim results every 100 seeds
            if (((seed+1) % 100) == 0):
                results_all.to_csv(os.getcwd().removesuffix('bash_scripts') + '/results/'+ str(date.today()) + '_FullCPMultistepDesignExpts_' + dataset + '_' + muh + '_nInit' + str(n_train_initial) + '_steps' + str(n_steps) + '_nseed' + str(n_seed) + '_lmbda' + str(lmbda) + '_reg' + str(reg) + '_depth' + str(depth_max) + '.csv',index=False)

    ## Save final results
    results_all.to_csv(os.getcwd().removesuffix('bash_scripts') + '/results/'+ str(date.today()) + '_FullCPMultistepDesignExpts_' + dataset + '_' + muh + '_nInit' + str(n_train_initial) + '_steps' + str(n_steps) + '_nseed' + str(n_seed) + '_lmbda' + str(lmbda) + '_reg' + str(reg) + '_depth' + str(depth_max) + '.csv',index=False)

    timestamp = time.time()
    value = datetime.datetime.fromtimestamp(timestamp)
    print("end time : ", value.strftime('%Y-%m-%d %H:%M:%S'))
