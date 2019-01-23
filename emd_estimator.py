#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 13:42:42 2018

@author: zeynep
"""

import numpy as np
from scipy import spatial, stats
#from model import tools
import time
import copy


import operator

from importlib import reload
import file_tools
reload(file_tools)
import data_tools
reload(data_tools)
import preferences
reload(preferences)
import constants
reload(constants)



class EMD_estimator():
    """
    Global estimator does not make instantaneous decisions or compute 
    instantaneous propbabilities. 
    
    The decision/probability is computed after the entire trajecory is observed. 
    It bags all observations form a single dyad so temporal relation is not 
    utilized.
    """
    def __init__(self):
        self.train_pdfs = {}
        self.train_histograms = {}
        
        self.test_pdfs = {}
        self.test_histograms = {}
        
        self.data_fnames = file_tools.get_data_fnames('../data/gender_compositions/')
        
        
    def init_conf_mat(self):
        """
        Initialize confusion matrix 
        """
        self.conf_mat = {}
        self.conf_mat['trajectory_based'] = {} # single key
        self.confidence = [] # hack for writing the data to txt file
            
        for class_gt in preferences.CLASSES:     
            self.conf_mat['trajectory_based'][class_gt] = {}            
            for class_est in preferences.CLASSES:
                self.conf_mat['trajectory_based'][class_gt][class_est] = 0

        if preferences.HIERARCHICAL is 'stage1':
            temp_conf_mat_dim1 = preferences.CLASSES_RAW
            self.conf_mat['trajectory_based_with_gt_fund'] = {}
            for class_gt in temp_conf_mat_dim1:
                self.conf_mat['trajectory_based_with_gt_fund'][class_gt] = {}
                for class_est in preferences.CLASSES:
                    self.conf_mat['trajectory_based_with_gt_fund'][class_gt][class_est] = 0
                    
               

    def set_train_dists(self, train_fnames):
        
        # initialize empty histograms
        # since histogram is accumulated as below, it needs to be initialized 
        # at every training           
        for c in preferences.CLASSES:
            self.train_histograms[c] = {}
            self.train_pdfs[c] = {}
            for o in preferences.OBSERVABLES:
                self.train_histograms[c][o] = data_tools.initialize_histogram(o)
                                
        # compute histograms for each class (using training set)
        for c in preferences.CLASSES:   
            for train_fname in train_fnames[c]:
                
                data = np.load(train_fname)
                data_A, data_B = data_tools.extract_individual_data(data)
                obs_data = data_tools.compute_observables(data_A, data_B)
                
                for o in preferences.OBSERVABLES:
                    self.train_histograms[c][o] += data_tools.compute_histogram_1D(o, obs_data[o])
                    
        for c in preferences.CLASSES:
            for o in preferences.OBSERVABLES:
                self.train_pdfs[c][o] = data_tools.compute_pdf(o, self.train_histograms[c][o])


    
    def set_test_dists(self, test_fnames):
        
        # initialize empty histograms
        # since one histogram/pdf  is computed for each element of test set
        # as below, it needs to be initialized at every testing
        for c in preferences.CLASSES:
            self.test_histograms[c], self.test_pdfs[c] = {}, {}
            for test_fname in test_fnames[c]:
                self.test_histograms[c][test_fname], self.test_pdfs[c][test_fname] = {}, {}
                for o in preferences.OBSERVABLES:
                    self.test_histograms[c][test_fname][o] = data_tools.initialize_histogram(o)
                    self.test_pdfs[c][test_fname][o] = []
                
        # compute histograms for each class (using test set)
        for c in preferences.CLASSES:   
            for test_fname in test_fnames[c]:

                data = np.load(test_fname)
                data_A, data_B = data_tools.extract_individual_data(data)
                obs_data = data_tools.compute_observables(data_A, data_B)
                
                for o in preferences.OBSERVABLES:
                    self.test_histograms[c][test_fname][o] = data_tools.compute_histogram_1D(o, obs_data[o])

        for c in preferences.CLASSES:
            for test_fname in test_fnames[c]:
                for o in preferences.OBSERVABLES:
                    self.test_pdfs[c][test_fname][o] = data_tools.compute_pdf(o, self.test_histograms[c][test_fname][o])

                
    def get_EMD_single(self, pdfA, pdfB):
        """
        Computes earth mover distance between two pdfs of a SINGLE (the same kind 
        of observable). Obviously, it is symmetric.
        
        Careful that the pdf's are normalized to sum up to 1, and **NOT** the 
        integral of the pdf is 1.
        
        The order of inputs does not matter
        """
       
        emd = []
        emd.append(0);
        
        if((np.sum(pdfA) - np.sum(pdfB)) > 2.220**(-10) ):
            print('sum(pdf1) = {0:.5f}  sum(pdf2) = {0:.5f} Make sure arrays are scaled'.format(\
                  (np.sum(pdfA),  np.sum(pdfB))) )
            return 0
                
        for i  in range(0, len(pdfA)):
            emd.append(pdfA[i] + emd[i] - pdfB[i])
        
        emd = np.sum(np.abs(emd))
        
        return emd

    def get_EMD_joint(self, pjA, pjB):
        """
        Computes earth mover distance between two sets of pdfs. Therefore, I call 
        it joint. Obviously, it is symmetric and the order of inputs does not 
        matter.
        
        Note that I use pdf's scaling up to 1 (and  **NOT** the integral)
        
        But then, in order to have a value as independent as the number of bins, I 
        scale the components with the associated bin_size.
        
        Due to the assumtion of independence of obsevables, I sum up the divergence 
        along each dimension. 
        
        """
        div_symmetric_pjB2pjA = 0
        
        for o1 in preferences.OBSERVABLES:
            
            tempA = pjA[o1] / np.sum(pjA[o1])
            tempB = pjB[o1] / np.sum(pjB[o1])
            (min_bound, max_bound, bin_size) = constants.HISTOG_PARAM_TABLE[o1]
            n_bins = np.abs(max_bound-min_bound) / bin_size
      
            div_sub  = self.get_EMD_single(tempA, tempB) / n_bins
            
            div_symmetric_pjB2pjA += div_sub
    
        return div_symmetric_pjB2pjA

    def estimate(self, test_fnames):
        """
        Performance is evaluated in only one way: assigning the dyad to the social
        relation class which lies in closest distance and building a confusion 
        matrix from such decisions.
        """

        for class_gt in preferences.CLASSES:
            for test_fname in test_fnames[class_gt]:
                
                class_gt_fund = test_fname.split('/')[3]
             
                # the following involve an array for each observable
                test_pdfs = self.test_pdfs[class_gt][test_fname]            
                     
                distances  = {} 

                for class_query in preferences.CLASSES:
                    
                    # the following involve an array for each observable
                    train_pdfs = self.train_pdfs[class_query]
                
                    distances[class_query] = \
                        self.get_EMD_joint(train_pdfs, test_pdfs)
                    
                    
                class_est = min(distances.items(), key=operator.itemgetter(1))[0]
                
                self.conf_mat['trajectory_based'][class_gt][class_est] += 1
                if preferences.HIERARCHICAL is 'stage1':
                    self.conf_mat['trajectory_based_with_gt_fund'][class_gt_fund][class_est] += 1
                                            
                # IMPORTANT:
                # Conf_mat needs to be scaled such that each row adds up
                # to 1. This will be done by the scale_conf_mats function
                ###############################################################

    def scale_conf_mats(self):
        """
        Scales the confusion matrices such that all rows add up to 1.
        """
        
        self.conf_mat_not_scaled = copy.deepcopy(self.conf_mat.copy())
        
        if preferences.HIERARCHICAL is 'stage1':
            temp_conf_mat_dim1 = preferences.CLASSES_RAW
        else:
            temp_conf_mat_dim1 = preferences.CLASSES
            
        keys = self.conf_mat.keys()
        for key in keys:
            if 'with_gt_fund' in key:
                for class_gt in temp_conf_mat_dim1:
                    factor = 0
                    for class_est in preferences.CLASSES:
                        factor += self.conf_mat[key][class_gt][class_est]
                        
                    for class_est in preferences.CLASSES:
                        self.conf_mat[key][class_gt][class_est] = self.conf_mat[key][class_gt][class_est] / factor
            elif 'with_gt_fund' not in key:
                for class_gt in preferences.CLASSES:
                    factor = 0
                    for class_est in preferences.CLASSES:
                        factor += self.conf_mat[key][class_gt][class_est]
                        
                    for class_est in preferences.CLASSES:
                        self.conf_mat[key][class_gt][class_est] = self.conf_mat[key][class_gt][class_est] / factor                
                                     
    
    def cross_validate(self):
        
        year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
        out_fname = 'results/'+ year +'_'+ month +'_'+ day +'_'+ hour +'_'+ minute + \
        '_emd' + \
        '_hier_' + (preferences.HIERARCHICAL) + '.txt'
        

        
        self.init_conf_mat()        
        for epoch in range(preferences.N_EPOCH):
            
            train_fnames, test_fnames = file_tools.shuffle_data_fnames(self.data_fnames)
            self.set_train_dists(train_fnames)
            self.set_test_dists(test_fnames)
            
            self.estimate(test_fnames)
            
        self.scale_conf_mats() 

                
        file_tools.write_conf_mat_to_file(out_fname, \
                                             'emd', \
                                             self.conf_mat, \
                                             self.conf_mat_not_scaled, \
                                             self.confidence, \
                                             alpha_val=[], \
                                             filtering_val=[], \
                                             measure_val='emd')
                    
                