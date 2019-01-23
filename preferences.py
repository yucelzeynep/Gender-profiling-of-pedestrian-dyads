#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 09:10:31 2018

@author: zeynep
"""

TRAIN_RATIO = 0.3
N_EPOCH = 20

#'bayesian', 'emd'
METHODS = ['bayesian']

# options are 'stage1', 'stage2', 'off'
HIERARCHICAL = 'off'

# class options are 'males', 'females', 'mixed'
CLASSES_RAW = [ 'males',  'females', 'mixed']
TARGET_CLASS = 'males'

# if hierarchical is off, options are 'koibito', 'doryo', 'yujin', 'kazoku', 'kazokuc', 'kazokua'
# if hierarchical is on, it is either stage 1 or stage2
# In stage1, options are doryo and others
# In stage2, options are the ones except doryo
if HIERARCHICAL is 'off':
    CLASSES = CLASSES_RAW.copy()
elif HIERARCHICAL is 'stage1':
    CLASSES = [TARGET_CLASS, 'others']
    OTHERS = [item for item in CLASSES_RAW if item not in CLASSES]
elif HIERARCHICAL is 'stage2':
    CLASSES = CLASSES_RAW.copy()
    CLASSES.remove(TARGET_CLASS)
    
# observable options are
# 'd', 'dx', 'dy',
# 'v_g', 'a_g',
# 'v_diff', 'vv_dot', 'vd_dot'
# 'h_diff', 'h_avg', 'h_short', 'h_diff', 'h_tall'
# or
#  'd', 'dx', 'dy', 'v_g', 'a_g', 'v_diff', 'vv_dot', 'vd_dot', 'h_avg', 'h_short', 'h_diff', 'h_tall'
    
OBSERVABLES = ['d', 'v_g' , 'v_diff']

# to be used in updating the probabiilty in bayesian approach
ALPHAS = [0.95]

