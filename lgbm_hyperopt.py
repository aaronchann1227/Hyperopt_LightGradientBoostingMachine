import pandas as pd
import numpy as np
import math
from timeit import default_timer as timer
import random
from scipy import sparse
from sklearn import metrics
import zipfile
import os
import copy
import sys
import lightgbm as lgb


#######################################################################################################################

# ### SET paramter in this function

# paramters
MAX_EVALS = 0 # the sum of all trials we want to get including previous trials
N_FOLDS = 10 # cross validation parameter
global  ITERATION
ITERATION = 20 
#pre_trials = "mytrial_240.p" 
new_trials = "mytrial_20.p" # file to save new trials
out_file = 'gbm_trials.csv' # file to save result for each iteration


# Load the data
##################### Data #########################################################################################
data = pd.read_csv("training_Data.csv",low_memory=False)
X = data.drop(columns=['Y'])
y = data['Y']

x_train_sparse = sparse.csr_matrix(X) 

# Create a lgb dataset
train_set = lgb.Dataset(x_train_sparse, y)



# Define the objective function you need to optimize
################# track evals ######################################################################################
import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer


def objective(params, n_folds=N_FOLDS):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""

    # Keep track of evals
    global ITERATION

    ITERATION += 1

    # Retrieve the subsample if present otherwise set to 1.0
    subsample = params['boosting_type'].get('subsample', 1.0)

    # Extract the boosting type
    params['boosting_type'] = params['boosting_type']['boosting_type']
    params['subsample'] = subsample

    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples','max_depth']:
        params[parameter_name] = int(params[parameter_name])
    
    start = timer()

    # Perform n_folds cross validation
    cv_results = lgb.cv(params, train_set, num_boost_round=10000, nfold=n_folds,
                        early_stopping_rounds=100, 
                        #mse mae
                        metrics='mse',
                        #metrics = 'mae',
                        stratified=False ,seed=50)


    # Extract the best score
    best_score = np.min(cv_results['l2-mean'])
    # best_score = np.min(cv_results['l1-mean'])

    # Minimized Loss
    loss = best_score

    # Boosting rounds that returned the lowest cv score
    # n_estimators = int(np.argmin(cv_results['l2-mean']) )
    n_estimators = int(np.argmin(cv_results['l2-mean']) )
    
    run_time = timer() - start

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, n_estimators,run_time])

    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'estimators': n_estimators,
            'train_time': run_time,
            'status': STATUS_OK}


# Define parameter search space
###############################################################################################################
from hyperopt import hp
from hyperopt.pyll.stochastic import sample

# Define the search space
space = {
     'objective':'tweedie',
     'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.6, 1)},
                                                 #{'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                                 #{'boosting_type': 'goss', 'subsample': 1.0}
                                                ]),
    'feature_fraction': hp.uniform('feature_fraction', 0.1, 1.0),
   # 'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)，
    'num_leaves': hp.quniform('num_leaves', 30, 100, 10),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.1)),
    'max_depth': hp.quniform('max_depth', 1, 10, 1),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 300, 10),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0)
}

from hyperopt import tpe
# optimization algorithm
tpe_algorithm = tpe.suggest
###############################################################################################################

from hyperopt import Trials
# Keep track of results
import pickle

#save trials so that we can continue tunning  
bayes_trials = Trials()  #initial trial
#bayes_trials = pickle.load(open(pre_trials, "rb"))  #load previous trials


# Connect to the text file to record the process
####################first trail######################################################
#initial saving in a new file
of_connection = open(out_file, 'w')  
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators','run_time']) 
of_connection.close()
#######################################################################################

from hyperopt import fmin
# Run optimization
best = fmin(fn = objective, space = space, algo = tpe.suggest,
            max_evals = MAX_EVALS, trials = bayes_trials, rstate = np.random.RandomState(50))

pickle.dump(bayes_trials, open(new_trials, "wb"))

########################################################################################################################



























