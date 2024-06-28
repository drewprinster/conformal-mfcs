#!/bin/sh
#SBATCH -t 54:00:00 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --mem=5G 
python ../run_FullCP_MFCS_expts.py --fitness_str red --n_train_initial 32 --lmbdas 8 --n_seed 1000 --n_steps 5 --reg 0.01 --depth_max 2
