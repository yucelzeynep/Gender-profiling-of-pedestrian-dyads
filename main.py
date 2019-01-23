#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 09:12:18 2018

@author: zeynep
"""

import bayesian_estimator
import emd_estimator
#import global_model_indep 
#import global_estimators_ND 

import numpy as np
import time

# if the constants/preferences are not refreshed, take a long way and reload
import preferences
from importlib import reload
reload(preferences)


if __name__ == "__main__":

    start_time = time.time()
    
    for method in preferences.METHODS:
        if method is 'bayesian':
            bayesian = bayesian_estimator.BayesianEstimator()
            bayesian.cross_validate()       
        elif(method is 'emd'):
            emd = emd_estimator.EMD_estimator()
            emd.cross_validate()

        
    elapsed_time = time.time() - start_time
    print('\nTime elapsed %2.2f sec' %elapsed_time)


    