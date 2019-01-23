#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 09:42:36 2018

@author: zeynep
"""
import numpy as np 
import copy
import random


from pathlib import Path
import pickle
from os import listdir

from importlib import reload

import constants
reload(constants)
import preferences
reload(preferences)

def load_gender_gt():
    gt = []
    
    for c in constants.CLASSES:
        temp = constants.GT_FPATH 
        fpaths = sorted([temp + f for f in listdir(temp) if '.dat' in f and 'st' in f])   

        for fpath in fpaths:
                data = np.loadtxt(fpath)
                
                for pt in data:
                    
                    temp = {'idA':[], 'idB': [], 'gender': []}
                    
                    [idA, idB] = sorted(pt[0:2]) 
                    temp['idA'] = int(idA)
                    temp['idB'] = int(idB)
                    
                    if int(pt[6]) == 0 : # number of males is 0                 
                        temp['gender'] = 'females' # all female
                    elif int(pt[6]) == 1 : # number of males is 1, so the other is female     
                        temp['gender'] = 'mixed' # mixed gender
                    elif int(pt[6]) == 2 : # number of males is 2, so no female     
                        temp['gender'] = 'males' # all male
                    else:
                        print('Problem with {} and {}'.format(idA, idB))
                        break
                    gt.append(temp)
                    
                    
    return gt
                    
                        
def saveas_wrt_gender(gt):
    """
    I use the same file name and the file is saved under a different (gender) 
    folder.
    
    Since some files under different social relation folders may have the same name, 
    when they are saved under gender folders, I may need to add a _v2 or _v3 at the 
    end of the file names.
    """

    for c in constants.CLASSES:
        temp = constants.SOCIAL_RELATION_FPATH + c +'/'
        fpaths = sorted([temp + f for f in listdir(temp) if '.dat' in f and 'threshold' in f])   

        for fpath in fpaths:
            
            data = np.load(fpath)
            
            ids = set(data[:,1])
            idA = min(ids)
            idB = max(ids)
            
            temp = next(item for item in gt if ((item['idA'] == idA and item['idB'] == idB) or(item['idA'] == idB and item['idB'] == idA)))
            
            if len(temp):
                
                fpath_new = copy.copy(fpath)
                fpath_new = fpath_new.replace('classes', 'gender_compositions')
                
                fpath_new = fpath_new.replace('doryo', temp['gender'] )
                fpath_new = fpath_new.replace('kazokua', temp['gender'] )
                fpath_new = fpath_new.replace('yujin', temp['gender'] )
                fpath_new = fpath_new.replace('koibito', temp['gender'] )
                
                my_file = Path(fpath_new)
                if my_file.is_file():
                    fpath_new = fpath_new.replace('.dat', '_v2.dat' )
                    
                    my_file = Path(fpath_new)
                    if my_file.is_file():
                        fpath_new = fpath_new.replace('_v2.dat', '_v3.dat' )
                
                with open(fpath_new, 'wb') as outfile:
                    pickle.dump(data, outfile)
           
                
   

def get_data_fnames(data_path):
    """
    Get the dataset for the given classes
    """
    data_fnames = {}
    for c in preferences.CLASSES_RAW:
        class_path = data_path + c +'/'
        class_set = [class_path + f for f in listdir(class_path) if 'threshold' in f]
        data_fnames[c] = class_set
    return data_fnames

def shuffle_data_fnames(data_fnames):
    """
    Randomly partition the data_fnames into training and testing sets
    """
    train_fnames, test_fnames = {}, {}
    train_fnames['others'] = []
    test_fnames['others'] = []
            
    for c, data_fname in data_fnames.items():
        n = len(data_fname)
        n_train = round(preferences.TRAIN_RATIO * n)
        shuffled_set = random.sample(data_fname, n)

        if preferences.HIERARCHICAL is 'off' or preferences.HIERARCHICAL is 'stage2':
            
            train_fnames[c] = shuffled_set[:n_train]
            test_fnames[c] = shuffled_set[n_train:]
            
        elif preferences.HIERARCHICAL is 'stage1':

            if c is preferences.TARGET_CLASS:
                train_fnames[c] = shuffled_set[:n_train]
                test_fnames[c] = shuffled_set[n_train:]
            else:
                train_fnames['others'].extend( shuffled_set[:n_train] )
                test_fnames['others'].extend( shuffled_set[n_train:] )
            
    return train_fnames, test_fnames


        
def write_conf_mat_to_file(out_fname, method, conf_mat, conf_mat_not_scaled, confidence, alpha_val, filtering_val, measure_val):
        
    if preferences.HIERARCHICAL is 'stage1':
        temp_conf_mat_dim1 = preferences.CLASSES_RAW
    else:
        temp_conf_mat_dim1 = preferences.CLASSES
    
    with open(out_fname, "a") as myfile:
        
        myfile.write('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
        myfile.write(method + '\n')
        myfile.write('Number of epochs: {}\n'.format(preferences.N_EPOCH))
        myfile.write('Train ratio: {}\n'.format(preferences.TRAIN_RATIO))
        myfile.write('Hierarchical: {}\n'.format(preferences.HIERARCHICAL))
        myfile.write('Observables: {}\n'.format(preferences.OBSERVABLES))

        if preferences.HIERARCHICAL is 'stage1':
            myfile.write('Others: {}\n'.format(preferences.OTHERS))
        
        if "bayesian" in method: 
            if "_dep" in method: 
                myfile.write('Filtering: {}\n'.format(filtering_val))            
            myfile.write('Alpha: {}\n'.format(alpha_val))
        elif "global" in method: 
            myfile.write('Measure: {}\n'.format(measure_val))
        ###################################################################
        ###################################################################
        if "bayesian" in method: 
            myfile.write('\n-----------------------------------\n')

            myfile.write('\nbinary_by_event: \n\t')
            for c in preferences.CLASSES:               
                myfile.write('{}\t'.format(c))
    
            
            for c_gt in preferences.CLASSES:
                myfile.write('\n{}\t'.format(c_gt))
                for c_est in preferences.CLASSES:
                    myfile.write('{:.2f}\t'.format(\
                          conf_mat['binary_by_event'][c_gt][c_est]*100))
            
            # compute perf cum
            s, d = 0,0
            keys = conf_mat_not_scaled['binary_by_event'].keys()
            for key1 in keys:
                for key2 in keys:
                    s += conf_mat_not_scaled['binary_by_event'][key1][key2]
                    if key1 is key2:
                        d += conf_mat_not_scaled['binary_by_event'][key1][key2]
            myfile.write('\nCum:\t{:.2f}\t'.format(d/s*100))
                    
            if preferences.HIERARCHICAL is 'stage1':   
                myfile.write('\n\nbinary_by_event_with_gt_fund: \n\t')
                for c in preferences.CLASSES:               
                    myfile.write('{}\t'.format(c))   
                    
                for c_gt in temp_conf_mat_dim1:
                    myfile.write('\n{}\t'.format(c_gt))
                    for c_est in preferences.CLASSES:
                        myfile.write('{:.2f}\t'.format(\
                              conf_mat['binary_by_event_with_gt_fund'][c_gt][c_est]*100))
        
#            ###################################################################
#       
#            myfile.write('\n-----------------------------------\n')
#            
#            myfile.write('\nbinary_by_trajectory_voting: \n\t')
#            for c in preferences.CLASSES:               
#                myfile.write('{}\t'.format(c))
#            
#            for c_gt in preferences.CLASSES:  
#                myfile.write('\n{}\t'.format(c_gt))
#                for c_est in preferences.CLASSES:      
#                    myfile.write('{:.2f}\t'.format(\
#                          conf_mat['binary_by_trajectory_voting'][c_gt][c_est]*100))
#
#            if preferences.HIERARCHICAL is 'stage1': 
#                myfile.write('\n\nbinary_by_trajectory_voting_with_gt_fund: \n\t')
#                for c in preferences.CLASSES:               
#                    myfile.write('{}\t'.format(c))   
#                    
#                for c_gt in temp_conf_mat_dim1:  
#                    myfile.write('\n{}\t'.format(c_gt))
#                    for c_est in preferences.CLASSES:      
#                        myfile.write('{:.2f}\t'.format(\
#                              conf_mat['binary_by_trajectory_voting_with_gt_fund'][c_gt][c_est]*100))
#        
            ###################################################################
            myfile.write('\n-----------------------------------\n')
            
            myfile.write('\nbinary_by_trajectory_probability: \n\t')
            for c in preferences.CLASSES:               
                myfile.write('{}\t'.format(c))
                
            for c_gt in preferences.CLASSES:  
                myfile.write('\n{}\t'.format(c_gt))
                for c_est in preferences.CLASSES:      
                    myfile.write('{:.2f}\t'.format(\
                          conf_mat['binary_by_trajectory_probability'][c_gt][c_est]*100))
                    
            # compute perf cum
            s, d = 0,0
            keys = conf_mat_not_scaled['binary_by_trajectory_probability'].keys()
            for key1 in keys:
                for key2 in keys:
                    s += conf_mat_not_scaled['binary_by_trajectory_probability'][key1][key2]
                    if key1 is key2:
                        d += conf_mat_not_scaled['binary_by_trajectory_probability'][key1][key2]
            myfile.write('\nCum:\t{:.2f}\t'.format(d/s*100))
                    
            if preferences.HIERARCHICAL is 'stage1': 
                myfile.write('\n\nbinary_by_trajectory_probability_with_gt_fund: \n\t')
                for c in preferences.CLASSES:               
                    myfile.write('{}\t'.format(c))
                
                for c_gt in temp_conf_mat_dim1:  
                    myfile.write('\n{}\t'.format(c_gt))
                    for c_est in preferences.CLASSES:      
                        myfile.write('{:.2f}\t'.format(\
                              conf_mat['binary_by_trajectory_probability_with_gt_fund'][c_gt][c_est]*100))
                    
#            ###################################################################
#            myfile.write('\n-----------------------------------\n')
#            myfile.write('\nprobabilistic_by_event: \n')
#            myfile.write('Confusion matrix for probabilistic_by_event\n\t')
#    
#            for c in preferences.CLASSES:               
#                myfile.write('{}\t'.format(c))
#    
#            for c_gt in preferences.CLASSES:  
#                myfile.write('\n{}\t'.format(c_gt))
#                for c_est in preferences.CLASSES:      
#                    myfile.write('{:.2f}\t'.format(\
#                          conf_mat['probabilistic_by_event'][c_gt][c_est]*100))
#
#            if preferences.HIERARCHICAL is 'stage1':  
#                myfile.write('\n\nprobabilistic_by_event_with_gt_fund: \n')
#                myfile.write('Confusion matrix for probabilistic_by_event_with_gt_fund\n\t')                
#                for c in preferences.CLASSES:               
#                    myfile.write('{}\t'.format(c))
#                    
#                for c_gt in temp_conf_mat_dim1:  
#                    myfile.write('\n{}\t'.format(c_gt))
#                    for c_est in preferences.CLASSES:      
#                        myfile.write('{:.2f}\t'.format(\
#                              conf_mat['probabilistic_by_event_with_gt_fund'][c_gt][c_est]*100))     
#            
#            myfile.write('\n\nConfidence for probabilistic_by_event\t')
#            Atot = 0
#            Btot = 0
#            Ntot = 0
#            for c_gt in preferences.CLASSES: 
#                
#                myfile.write('\n{}\t'.format(c_gt))
#                
#                A = confidence['probabilistic_by_event'][c_gt]['cum_confidence']
#                B = confidence['probabilistic_by_event'][c_gt]['cum_confidence_sq']
#                N = confidence['probabilistic_by_event'][c_gt]['n_observations']
#                
#                mu = A/N
#                sigma = np.sqrt(B/N - mu**2)
#                
#                myfile.write('{:.2f} pm {:.2f}\t'.format(\
#                             mu ,\
#                             sigma ))
#                
#                Atot += A
#                Btot += B
#                Ntot += N
#                
#            myfile.write('\n')
#    
#            mu_cum = Atot/Ntot
#            sigma_cum = np.sqrt(Btot/Ntot - mu_cum**2)
#            myfile.write('Tot\t{:.2f} pm {:.2f}\n'.format(mu_cum, sigma_cum))
#            ###################################################################
#            myfile.write('\n-----------------------------------\n')
#            
#            myfile.write('\nprobabilistic_by_trajectory: \n')
#            myfile.write('Confusion matrix for probabilistic_by_trajectory\n\t')
#    
#            for c in preferences.CLASSES:               
#                myfile.write('{}\t'.format(c))
#    
#            for c_gt in preferences.CLASSES:  
#                myfile.write('\n{}\t'.format(c_gt))
#                for c_est in preferences.CLASSES:      
#                    myfile.write('{:.2f}\t'.format(\
#                          conf_mat['probabilistic_by_trajectory'][c_gt][c_est]*100))
#
#            if preferences.HIERARCHICAL is 'stage1':  
#                myfile.write('\n\nprobabilistic_by_trajectory_with_gt_fund: \n')
#                myfile.write('Confusion matrix for probabilistic_by_trajectory_with_gt_fund\n\t')                
#                for c in preferences.CLASSES:               
#                    myfile.write('{}\t'.format(c))
#                
#                for c_gt in temp_conf_mat_dim1:  
#                    myfile.write('\n{}\t'.format(c_gt))
#                    for c_est in preferences.CLASSES:      
#                        myfile.write('{:.2f}\t'.format(\
#                              conf_mat['probabilistic_by_trajectory_with_gt_fund'][c_gt][c_est]*100))     
#            
#            myfile.write('\n\nConfidence for probabilistic_by_trajectory')
#            Atot = 0
#            Btot = 0
#            Ntot = 0
#            for c_gt in preferences.CLASSES: 
#                
#                myfile.write('\n{}\t'.format(c_gt))
#                
#                A = confidence['probabilistic_by_trajectory'][c_gt]['cum_confidence']
#                B = confidence['probabilistic_by_trajectory'][c_gt]['cum_confidence_sq']
#                N = confidence['probabilistic_by_trajectory'][c_gt]['n_observations']
#                
#                mu = A/N
#                sigma = np.sqrt(B/N - mu**2)
#                
#                myfile.write('{:.2f} pm {:.2f}\t'.format(\
#                             mu ,\
#                             sigma ))
#                
#                Atot += A
#                Btot += B
#                Ntot += N
#                
#            myfile.write('\n')
#    
#            mu_cum = Atot/Ntot
#            sigma_cum = np.sqrt(Btot/Ntot - mu_cum**2)
#            myfile.write('Tot\t{:.2f} pm {:.2f}\n'.format(mu_cum, sigma_cum))
#            ###################################################################
#            myfile.write('\n-----------------------------------\n')
#            
#            myfile.write('\nempirical_probability_by_trajectory: \n')
#            myfile.write('Confusion matrix for empirical_probability_by_trajectory\n\t')
#    
#            for c in preferences.CLASSES:               
#                myfile.write('{}\t'.format(c))
#    
#            for c_gt in preferences.CLASSES:  
#                myfile.write('\n{}\t'.format(c_gt))
#                for c_est in preferences.CLASSES:      
#                    myfile.write('{:.2f}\t'.format(\
#                          conf_mat['empirical_probability_by_trajectory'][c_gt][c_est]*100))
#
#            if preferences.HIERARCHICAL is 'stage1':  
#                myfile.write('\n\nempirical_probability_by_trajectory_with_gt_fund: \n')
#                myfile.write('Confusion matrix for empirical_probability_by_trajectory_with_gt_fund\n\t')                
#                for c in preferences.CLASSES:               
#                    myfile.write('{}\t'.format(c))
#                
#                for c_gt in temp_conf_mat_dim1:  
#                    myfile.write('\n{}\t'.format(c_gt))
#                    for c_est in preferences.CLASSES:      
#                        myfile.write('{:.2f}\t'.format(\
#                              conf_mat['empirical_probability_by_trajectory_with_gt_fund'][c_gt][c_est]*100))      
#            
#            myfile.write('\n\nConfidence for empirical_probability_by_trajectory')
#            Atot = 0
#            Btot = 0
#            Ntot = 0
#            for c_gt in preferences.CLASSES: 
#                
#                myfile.write('\n{}\t'.format(c_gt))
#                
#                A = confidence['empirical_probability_by_trajectory'][c_gt]['cum_confidence']
#                B = confidence['empirical_probability_by_trajectory'][c_gt]['cum_confidence_sq']
#                N = confidence['empirical_probability_by_trajectory'][c_gt]['n_observations']
#                
#                mu = A/N
#                sigma = np.sqrt(B/N - mu**2)
#                
#                myfile.write('{:.2f} pm {:.2f}\t'.format(\
#                             mu ,\
#                             sigma ))
#                
#                Atot += A
#                Btot += B
#                Ntot += N
#                
#            myfile.write('\n')
#    
#            mu_cum = Atot/Ntot
#            sigma_cum = np.sqrt(Btot/Ntot - mu_cum**2)
#            myfile.write('Tot\t{:.2f} pm {:.2f}\n'.format(mu_cum, sigma_cum))
#            ###################################################################    
        if "emd" in method: 
            myfile.write('\n-----------------------------------\n')

            myfile.write('\ntrajectory_based: \n\t')
            for c in preferences.CLASSES:               
                myfile.write('{}\t'.format(c))
    
            for c_gt in preferences.CLASSES:  
                myfile.write('\n{}\t'.format(c_gt))
                for c_est in preferences.CLASSES:      
                    myfile.write('{:.2f}\t'.format(\
                          conf_mat['trajectory_based'][c_gt][c_est]*100))
            
            # compute perf cum
            s, d = 0,0
            keys = conf_mat_not_scaled['trajectory_based'].keys()
            for key1 in keys:
                for key2 in keys:
                    s += conf_mat_not_scaled['trajectory_based'][key1][key2]
                    if key1 is key2:
                        d += conf_mat_not_scaled['trajectory_based'][key1][key2]
            myfile.write('\nCum:\t{:.2f}\t'.format(d/s*100))

            if preferences.HIERARCHICAL is 'stage1':  
                myfile.write('\n\ntrajectory_based_with_gt_fund: \n\t')
                for c in preferences.CLASSES:               
                    myfile.write('{}\t'.format(c))
                
                for c_gt in temp_conf_mat_dim1:  
                    myfile.write('\n{}\t'.format(c_gt))
                    for c_est in preferences.CLASSES:      
                        myfile.write('{:.2f}\t'.format(\
                              conf_mat['trajectory_based_with_gt_fund'][c_gt][c_est]*100))              


