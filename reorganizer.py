#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 10:11:05 2018

@author: zeynep
"""
import time
import file_tools

"""

This functions loads the data from under social relation folders,
and saves the same things once more, this time in a different folder wrt gender.

"""

if __name__ == "__main__":

    start_time = time.time()
    
    gt = file_tools.load_gender_gt()
    file_tools.saveas_wrt_gender(gt)
    
    elapsed_time = time.time() - start_time
    print('\nTime elapsed %2.2f sec' %elapsed_time)
