#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 09:12:49 2018

@author: zeynep
"""

import numpy as np
from scipy import spatial, stats
import time
import copy


#import matplotlib.pyplot as 
import operator

from importlib import reload
import preferences
reload(preferences)

import data_tools
reload(data_tools)
import file_tools
reload(file_tools)


class BayesianEstimator():
    """
    Bayes class for the estimator
    """
    def __init__(self):
        self.train_pdfs1D = {}
        self.data_fnames = file_tools.get_data_fnames('../data/gender_compositions/')
                
        
    def init_conf_mat(self):
        """
        Initialize confusion matrix or performance measures.
        """
        
        self.conf_mat = {\
                         'binary_by_event':{},\
                         'binary_by_trajectory_voting':{},\
                         'binary_by_trajectory_probability':{},\
                         'probabilistic_by_event':{},\
                         'probabilistic_by_trajectory':{},\
                         'empirical_probability_by_trajectory':{}\
                         }
        
        self.confidence = {\
                           'probabilistic_by_event':{},\
                           'probabilistic_by_trajectory':{},\
                           'empirical_probability_by_trajectory':{}\
                           }
        
        for class_gt in preferences.CLASSES:

            self.conf_mat['binary_by_event'][class_gt] = {}
            self.conf_mat['binary_by_trajectory_voting'][class_gt] = {}
            self.conf_mat['binary_by_trajectory_probability'][class_gt] = {}
            self.conf_mat['probabilistic_by_event'][class_gt] = {}
            self.conf_mat['probabilistic_by_trajectory'][class_gt] = {}
            self.conf_mat['empirical_probability_by_trajectory'][class_gt] = {}

            self.confidence['probabilistic_by_event'][class_gt] = {\
                         'n_observations': 0, \
                         'cum_confidence': 0,\
                         'cum_confidence_sq': 0}
            self.confidence['probabilistic_by_trajectory'][class_gt] = {\
                         'n_observations': 0, \
                         'cum_confidence': 0,\
                         'cum_confidence_sq': 0}
            self.confidence['empirical_probability_by_trajectory'][class_gt] = {\
                         'n_observations': 0, \
                         'cum_confidence': 0,\
                             'cum_confidence_sq': 0}
                           

            for class_est in preferences.CLASSES:
                
                ##########################
                # new conf_mat keys
                self.conf_mat['binary_by_event'][class_gt][class_est] = 0
                self.conf_mat['binary_by_trajectory_voting'][class_gt][class_est] = 0
                self.conf_mat['binary_by_trajectory_probability'][class_gt][class_est] = 0
                self.conf_mat['probabilistic_by_event'][class_gt][class_est] = 0
                self.conf_mat['probabilistic_by_trajectory'][class_gt][class_est] = 0
                self.conf_mat['empirical_probability_by_trajectory'][class_gt][class_est] = 0
                
        # some additinal conf mats for hier stage1
        if preferences.HIERARCHICAL is 'stage1':
            temp_conf_mat_dim1 = preferences.CLASSES_RAW
            
            # additional keys
            self.conf_mat['binary_by_event_with_gt_fund'] = {}
            self.conf_mat['binary_by_trajectory_voting_with_gt_fund'] = {}
            self.conf_mat['binary_by_trajectory_probability_with_gt_fund'] = {}
            self.conf_mat['probabilistic_by_event_with_gt_fund'] = {}
            self.conf_mat['probabilistic_by_trajectory_with_gt_fund'] = {}
            self.conf_mat['empirical_probability_by_trajectory_with_gt_fund'] = {}

            for class_gt in temp_conf_mat_dim1:
    
                self.conf_mat['binary_by_event_with_gt_fund'][class_gt] = {}
                self.conf_mat['binary_by_trajectory_voting_with_gt_fund'][class_gt] = {}
                self.conf_mat['binary_by_trajectory_probability_with_gt_fund'][class_gt] = {}
                self.conf_mat['probabilistic_by_event_with_gt_fund'][class_gt] = {}
                self.conf_mat['probabilistic_by_trajectory_with_gt_fund'][class_gt] = {}
                self.conf_mat['empirical_probability_by_trajectory_with_gt_fund'][class_gt] = {}
    
                for class_est in preferences.CLASSES:
                    
                    ##########################
                    # new conf_mat keys
                    self.conf_mat['binary_by_event_with_gt_fund'][class_gt][class_est] = 0
                    self.conf_mat['binary_by_trajectory_voting_with_gt_fund'][class_gt][class_est] = 0
                    self.conf_mat['binary_by_trajectory_probability_with_gt_fund'][class_gt][class_est] = 0
                    self.conf_mat['probabilistic_by_event_with_gt_fund'][class_gt][class_est] = 0
                    self.conf_mat['probabilistic_by_trajectory_with_gt_fund'][class_gt][class_est] = 0
                    self.conf_mat['empirical_probability_by_trajectory_with_gt_fund'][class_gt][class_est] = 0


    def train(self, train_fnames):
        
        train_histograms1D = {}
        # initialize empty histograms
        for o in preferences.OBSERVABLES:
            train_histograms1D[o], self.train_pdfs1D[o] = {}, {}
            for c in preferences.CLASSES:
                train_histograms1D[o][c] = data_tools.initialize_histogram(o)
                
        # compute histograms for each class
        for c in preferences.CLASSES:   
            for file_path in train_fnames[c]:
                data = np.load(file_path)
                data_A, data_B = data_tools.extract_individual_data(data)
                obs_data = data_tools.compute_observables(data_A, data_B)
                for o in preferences.OBSERVABLES:
                    train_histograms1D[o][c] += data_tools.compute_histogram_1D(o, obs_data[o])
                    
        for o in preferences.OBSERVABLES:
            for c in preferences.CLASSES:
                self.train_pdfs1D[o][c] = data_tools.compute_pdf(o, train_histograms1D[o][c])

                
    def compute_probabilities(self, bins, alpha):
        
        n_data = len(bins[preferences.OBSERVABLES[0]])
        p_posts, p_prior, p_likes, p_conds = {}, {}, {}, {}
        
        for c in preferences.CLASSES:
            p_prior[c] = 1 / len(preferences.CLASSES)
            p_posts[c] = np.zeros((n_data))
            p_posts[c][0] = p_prior[c]
            
        for j in range(1, n_data):
            for c in preferences.CLASSES:
                p_likes[c] = 1
                for o in preferences.OBSERVABLES:
#                    if self.train_pdfs1D[o][c][bins[o][j]-1] > 0:
#                        p_likes[c] *= self.train_pdfs1D[o][c][bins[o][j]-1]
                    if bins[o][j] < len(self.train_pdfs1D[o][c]): #constants.NBINS_PARAM_TABLE[o]
                        p_likes[c] *= self.train_pdfs1D[o][c][bins[o][j]]

                p_prior[c] = (1 - alpha) * p_posts[c][0] + alpha * p_posts[c][j-1]
                p_conds[c] = p_likes[c] * p_prior[c]  
                
            s = sum(p_conds.values())
            # if an observation does not appear inof the classes, they will all 
            # get a 0 probability so the scaling below will give nan.
            # To avoid nans, I assume equal posteriors, which makes sense
            if s >0:
                for c in preferences.CLASSES:
                    p_posts[c][j] = p_conds[c] / s 
            else:
                for c in preferences.CLASSES:
                    p_posts[c][j] = 1 / len(preferences.CLASSES)
                        
                
        return p_posts

    def estimate(self, alpha, test_fnames):
        
        """
        
        Performance is evaluated in various ways. Below I explain each of these
        on a toy example.
        
        Lets assume we have a dyad whic nis annotated to be Doryo (D) and its 
        trajectory of length 8. Suppose the post probabilities are found 
        as follows:
            
            time = t [K     D     Y     Kz  ]
            time = 0 [0.45  0.20  0.10  0.25]
            time = 1 [0.20  0.10  0.45  0.25]
            time = 2 [0.45  0.20  0.10  0.25]
            time = 3 [0.20  0.45  0.10  0.25]
            time = 4 [0.45  0.20  0.10  0.25]
            time = 5 [0.25  0.20  0.10  0.45]
            time = 6 [0.45  0.20  0.10  0.25]
            time = 7 [0.20  0.45  0.10  0.25]
           
        Here, each vector involves (post) probabilities for koibito (K), 
        doryo (D), yujin (Y), and kazoku (Kz), respectively.
            
        -----------------------------------------------------------------------
        trajectory-based decision: yields a single estimation of social 
        relation for each dyad. 
        
        TODO
            
        -----------------------------------------------------------------------      
        binary_by_event
        
        For each trajectory data point (event), we make an 
        instantaneous decision by choosing the social relation class with highest 
        post probability.
        
        Then the instantaneous decisions will be:
                [K Y K D K Kz K D]
                
        We build a confusion matrix such that the rows represent the true class
        and the columns represent the assigned class. The above dyad contributes 
        to the confusion matrix (before scaling) as follows:
                                               
                                    [K D Y Kz ]
                                    | 0 0 0 0 |
         old_mats_before_scaling +  | 4 2 1 1 |
                                    | 0 0 0 0 |
                                    | 0 0 0 0 |
                                    
        After processing all dyads, each row is scaled to 1.
        -----------------------------------------------------------------------
        binary_by_trajectory_voting
        
        According to this assessment method, for each dyad we pick the social 
        relation class with highest number of votes among all trajectory data
        points. So eventually the dyad gets a single estimation (discrete output)
        
        For the above case, the votes are as follows:
            K = 4
            D = 2
            Y = 1
            Kz= 1
            
        So the estimated class will be K. If this decision is correct, it 
        will give a 1, otherwise a 0. Actually in the confusion matrix, I also 
        store the exact mistakes (off-diagonal).
        
        The above dyad contributes to the confusion matrix (before scaling) as 
        follows:
                                    [K D Y Kz ]
                                    | 0 0 0 0 |
         old_mats_before_scaling +  | 1 0 0 0 |
                                    | 0 0 0 0 |
                                    | 0 0 0 0 |
                                    
        After processing all dyads, each row is scaled to 1.        
        -----------------------------------------------------------------------
        binary_by_trajectory_probability
        
        For every dyad, we take the average probability concering each social 
        relation class over all trajectory data points. The class with the 
        highest probability is the esmtimated class.
        
        For the above case, the average probabilities are:
            
            K = 0.33125
            D = 0.25
            Y = 0.14375
            Kz= 0.275
        
        So the decision will be K.
            
        The above dyad contributes to the confusion matrix (before scaling) as 
        follows:
                                    [K D Y Kz ]
                                    | 0 0 0 0 |
         old_mats_before_scaling +  | 1 0 0 0 |
                                    | 0 0 0 0 |
                                    | 0 0 0 0 |

        After processing all dyads, each row is scaled to 1.        
        -----------------------------------------------------------------------
        probabilistic_by_event
        
        This assessment method yields two outcomes: 
            1. Confusion matrix
            2. Confidence values
        
        1. Confusion matrix:
            I accumulate the post probabilites derived from each trajectory data 
            point of each dyad, in a confusion matrix. 
            
        The above dyad contributes to the confusion matrix (before scaling) as 
        follows:
                                     | 0     0     0     0   |
            old_mats_before_scaling +| 0.45  0.20  0.10  0.25| + 
                                     | 0     0     0     0   |
                                     | 0     0     0     0   |
                                     
                | 0     0     0     0   |   | 0     0     0     0   |
                | 0.20  0.10  0.45  0.25| + | 0.45  0.20  0.10  0.25| + ...
                | 0     0     0     0   |   | 0     0     0     0   |
                | 0     0     0     0   |   | 0     0     0     0   |
                
                | 0     0     0     0   |   
                | 0.20  0.45  0.10  0.25|
                | 0     0     0     0   |  
                | 0     0     0     0   |   

        After processing all dyads, each row is scaled to 1.        

        2. Confidence values:
            
            The confidence is defined as below:
                conf = 100 - abs(p_max - p_gt)
        
            Here p_max is the highest probability (among the probabiities associated 
            with each possible outcome (ie class)). On the other hand, p_gt is the 
            probability that is associated with the gt class.
        
            I compute confidence at each single observation point (ie trajectory 
            point) I do not store all these values. Instead, I store only the 
            variables necessary to compute statistics. Namely:            
                the number of observations
                the sum confidence values
                the sum of squares of confidence values
                
            For the above example, the confidence values will be
            
            time = 1 - (0.45 - 0.20 ) = 0.75
            time = 1 - (0.45 - 0.10 ) = 0.65
            time = 1 - (0.45 - 0.20 ) = 0.75
            time = 1 - (0.45 - 0.45 ) = 1
            time = 1 - (0.45 - 0.20 ) = 0.75
            time = 1 - (0.45 - 0.20 ) = 0.75
            time = 1 - (0.45 - 0.20 ) = 0.75
            time = 1 - (0.45 - 0.45 ) = 1
            
            and I will update the stored values as follows:
                number_of_observations += 8
                sum_confidence_values += 0.75 + 0.65 + 0.75 + 1 + 0.75 + 0.75 + 
                0.75 + 1
                sum_of_squares_of_confidence_values = 0.75**2 + 0.65**2 + 
                0.75**2 + 1**2 + 0.75**2 + 0.75**2 + 0.75**2 + 1**2
                
        -----------------------------------------------------------------------
        probabilistic_by_trajectory
        
        This assessment method yields two outcomes: 
            1. Confusion matrix
            2. Confidence values
            
        1. Confusion matrix:
            Remember from binary_by_trajectory_probability that we computed the 
            average probability concering each social relation class over all 
            trajectory data points.         
            
            For the above case, the average probabilities were:
                K = 0.33125
                D = 0.25
                Y = 0.14375
                Kz= 0.275
            
            The above dyad contributes to the confusion matrix (before scaling)
            as follows:
                                         | 0        0     0        0    |
                old_mats_before_scaling +| 0.33125  0.25  0.14375  0.275| 
                                         | 0        0     0        0    |
                                         | 0        0     0        0    |
                                         
            After processing all dyads, each row is scaled to 1.        

                
        2. Confidence values:
            I compute the confidece on the above probability vector:
                confidence = 1 - (0.33125 - 0.25)
                           = 0.91875

            and I update the stored values as follows:
                number_of_observations += 1
                sum_confidence_values += 0.91875
                sum_of_squares_of_confidence_values = 0.91875**2        
        -----------------------------------------------------------------------
        empirical_probability_by_trajectory

        This assessment method yields two outcomes: 
            1. Confusion matrix
            2. Confidence values
        
        1. Confusion matrix:
            Here, I first derive a probability vector from the votes given at 
            each data point.
            
            For the above case, the votes were as follows:
            K = 4
            D = 2
            Y = 1
            Kz= 1
            
            So I build a probability vector as:
                [K   D   Y   Kz ]
                [4/8 2/8 1/8 1/8]
                
            Then this vector is added to the confusion matrix as below
                                     | 0     0     0      0    |
            old_mats_before_scaling +| 0.50  0.25  0.125  0.125| 
                                     | 0     0     0      0    |
                                     | 0     0     0      0    |
                                     
            After processing all dyads, each row is scaled to 1.        

        2. Confidence values:

            The confidence is defined as below:
                conf = 100 - abs(p_max - p_gt)
            
            This time p_max is the highest value in the above vector. Similar to
            the previous case, p_gt is the probability associated with the gt 
            class.
            
            For the above case, since the gt class is given as D, conf will be:
                conf = 1 - abs(0.50 - 0.25)
                       = 0.75
                       
            I will update the stored values as follows:
                number_of_observations += 1
                sum_confidence_values += 0.75 
                sum_of_squares_of_confidence_values += 0.75**2 
                   
            Same as before, values close to 1 indicate that there is a mistake 
            but not that big.
        
        -----------------------------------------------------------------------
        """
    
        for class_gt in preferences.CLASSES:
                
            for test_fname in test_fnames[class_gt]:
                 
                 # fundamental gt, in case of hier stage 1 
                 # (while others contain 3 classes)
                class_gt_fund = test_fname.split('/')[3] # hardcoded
                
                data = np.load(test_fname)
                data_A, data_B = data_tools.extract_individual_data(data)
                N_observations = len(data_A) # len(data_B) is the same
                obs_data = data_tools.compute_observables(data_A, data_B)
                                
                bins = {}
                for o in preferences.OBSERVABLES:
                    bins[o] = data_tools.find_bins(o, obs_data[o])
                p_posts =  self.compute_probabilities(bins, alpha) 
                
                ###############################################################
                #
                # binary_by_event
                # 
                for i in range(0, N_observations):
                    # get all instantaneous probabilities
                    p_inst = {} #instantaneous probabilities
                    for class_temp in preferences.CLASSES:
                        p_inst[class_temp] = p_posts[class_temp][i]
                    
                    # class_est is the estimated class for this data point 
                    class_est = max(p_inst.items(), key=operator.itemgetter(1))[0]
                    self.conf_mat['binary_by_event'][class_gt][class_est] += 1
                    if preferences.HIERARCHICAL is 'stage1':
                        self.conf_mat['binary_by_event_with_gt_fund'][class_gt_fund][class_est] += 1
                            
                    
                # IMPORTANT:
                # This matrix needs to be scaled such that each row adds up
                # to 1. This will be done when I write the matrix to the 
                # txt data file
                ###############################################################
                #
                # binary_by_trajectory_voting
                #              
                n_votes = {}
                for class_temp in preferences.CLASSES:
                    n_votes[class_temp] = 0
                
                for i in range(0, N_observations):
                    # get all instantaneous probabilities
                    p_inst = {} #instantaneous probabilities
                    for class_temp in preferences.CLASSES:
                        p_inst[class_temp] = p_posts[class_temp][i]
                    
                    # one vote is given at every daya point
                    # the vote goes to the class with highest post prob
                    # class_est is the estimated class for this data point 
                    class_est = max(p_inst.items(), key=operator.itemgetter(1))[0]
                    n_votes[class_est] += 1
                    
                # the estimated class is the one which receives highest number 
                # of votes (along the trajectory)
                class_est_voting_winner = max(n_votes.items(), key=operator.itemgetter(1))[0]
                self.conf_mat['binary_by_trajectory_voting'][class_gt][class_est_voting_winner] += 1
                if preferences.HIERARCHICAL is 'stage1':
                    self.conf_mat['binary_by_trajectory_voting_with_gt_fund'][class_gt_fund][class_est_voting_winner] += 1
                                    
                # IMPORTANT:
                # This matrix needs to be scaled such that each row adds up
                # to 1. This will be done when I write the matrix to the 
                # txt data file
                ###############################################################
                #
                # binary_by_trajectory_probability
                #
                p_mean = {}
                for class_est in preferences.CLASSES:
                    # class_est is not really the 'output decision'
                    p_mean[class_est] = np.mean(p_posts[class_est])
                    
                p_max = max(p_mean.items(), key=operator.itemgetter(1))[1] 
                c_out = max(p_mean.items(), key=operator.itemgetter(1))[0] 
                self.conf_mat['binary_by_trajectory_probability'][class_gt][c_out] += 1
                if preferences.HIERARCHICAL is 'stage1':
                    self.conf_mat['binary_by_trajectory_probability_with_gt_fund'][class_gt_fund][c_out] += 1
                # IMPORTANT:
                # This matrix needs to be scaled such that each row adds up
                # to 1. This will be done when I write the matrix to the 
                # txt data file
                ###############################################################
                #
                # probabilistic_by_event
                # 
                for i in range(0, N_observations):
                    # get all instantaneous probabilities
                    p_inst = {} #instantaneous probabilities
                    for class_temp in preferences.CLASSES:
                        p_inst[class_temp] = p_posts[class_temp][i]
                        self.conf_mat['probabilistic_by_event'][class_gt][class_temp] += p_inst[class_temp]
                        if preferences.HIERARCHICAL is 'stage1':
                            self.conf_mat['probabilistic_by_event_with_gt_fund'][class_gt_fund][class_temp] += p_inst[class_temp]
                    
                    p_max = max(p_mean.items(), key=operator.itemgetter(1))[1] 
                    p_gt = p_inst[class_gt]
                    confidence = 1 - (p_max - p_gt)
                        
                    self.confidence['probabilistic_by_event'][class_gt]['n_observations'] += 1
                    self.confidence['probabilistic_by_event'][class_gt]['cum_confidence'] += confidence
                    self.confidence['probabilistic_by_event'][class_gt]['cum_confidence_sq'] += confidence**2
                # IMPORTANT:
                # Conf_mat needs to be scaled such that each row adds up
                # to 1. This will be done by the scale_conf_mats function
                # In addition, I will derive the statistics regarding confidence,
                # ie mean and std, from the three stored values
                ###############################################################
                #      
                # probabilistic_by_trajectory
                #
                p_mean = {}
                for class_est in preferences.CLASSES:
                    # class_est is not really the 'output decision'
                    p_mean[class_est] = np.mean(p_posts[class_est])
                    self.conf_mat['probabilistic_by_trajectory'][class_gt][class_est] += p_mean[class_est]
                    if preferences.HIERARCHICAL is 'stage1':
                        self.conf_mat['probabilistic_by_trajectory_with_gt_fund'][class_gt_fund][class_est] += p_mean[class_est]
                    
                p_max = max(p_mean.items(), key=operator.itemgetter(1))[1]
                p_gt = p_mean[class_gt]
                confidence = 1 - (p_max - p_gt)
                
                self.confidence['probabilistic_by_trajectory'][class_gt]['n_observations'] += 1
                self.confidence['probabilistic_by_trajectory'][class_gt]['cum_confidence'] += confidence
                self.confidence['probabilistic_by_trajectory'][class_gt]['cum_confidence_sq'] += confidence**2
                # IMPORTANT:
                # Conf_mat needs to be scaled such that each row adds up
                # to 1. This will be done by the scale_conf_mats function
                # In addition, I will derive the statistics regarding confidence,
                # ie mean and std, from the three stored values
                ###############################################################
                #      
                # empirical_probability_by_trajectory
                #
                n_votes = {}
                for class_temp in preferences.CLASSES:
                    n_votes[class_temp] = 0
                
                for i in range(0, N_observations):
                    # get all instantaneous probabilities
                    p_inst = {} #instantaneous probabilities
                    for class_temp in preferences.CLASSES:
                        p_inst[class_temp] = p_posts[class_temp][i]
                    
                    # one vote is given at every daya point
                    # the vote goes to the class with highest post prob
                    # class_est is the estimated class for this data point 
                    class_est = max(p_inst.items(), key=operator.itemgetter(1))[0]
                    n_votes[class_est] += 1
                
                # scale the votes to 1, such that they represent probabilities
                factor = 1.0/sum(n_votes.values())  
                class_est_emp_probs = {k: v*factor for k, v in n_votes.items() }
                
                for class_est in preferences.CLASSES:
                    # class_est is not really the 'output decision'
                    # here I only keep the probability associated with every 
                    # possible outcome
                    self.conf_mat['empirical_probability_by_trajectory'][class_gt][class_est] += \
                    class_est_emp_probs[class_est]
                    
                    if preferences.HIERARCHICAL is 'stage1':
                        self.conf_mat['empirical_probability_by_trajectory_with_gt_fund'][class_gt_fund][class_est] += \
                        class_est_emp_probs[class_est]                        

                
                
                p_max = max(class_est_emp_probs.items(), key=operator.itemgetter(1))[1] 
                p_gt = class_est_emp_probs[class_gt]
                confidence = 1 - (p_max - p_gt)
                
                self.confidence['empirical_probability_by_trajectory'][class_gt]['n_observations'] += 1
                self.confidence['empirical_probability_by_trajectory'][class_gt]['cum_confidence'] += confidence
                self.confidence['empirical_probability_by_trajectory'][class_gt]['cum_confidence_sq'] += confidence**2
               
                # IMPORTANT:
                # Conf_mat needs to be scaled such that each row adds up
                # to 1. This will be done by the scale_conf_mats function
                # In addition, I will derive the statistics regarding confidence,
                # ie mean and std, from the three stored values
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
        
        # output file name
        year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
        out_fname = 'results/'+ year +'_'+ month +'_'+ day +'_'+ hour +'_'+  minute + \
        '_bayesian_indep' + \
        '_hier_' + (preferences.HIERARCHICAL) + '.txt'


        for alpha in preferences.ALPHAS:
            
            self.init_conf_mat()

            for epoch in range(preferences.N_EPOCH):
                
                train_fnames, test_fnames = file_tools.shuffle_data_fnames(self.data_fnames)
                self.train(train_fnames)
                self.estimate(alpha, test_fnames)
                
            self.scale_conf_mats() 
                
            file_tools.write_conf_mat_to_file(out_fname, \
                                                 'bayesian_indep', \
                                                 self.conf_mat, \
                                                 self.conf_mat_not_scaled , \
                                                 self.confidence, \
                                                 alpha_val=alpha, \
                                                 filtering_val=[], \
                                                 measure_val=[])
                
 