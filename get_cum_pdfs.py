#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 13:47:24 2018

@author: zeynep
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:12:17 2018

@author: zeynep
"""
from importlib import reload

import data_tools
reload(data_tools)
import file_tools
reload(file_tools)
import constants
reload(constants)
import preferences
reload(preferences)

import matplotlib.pyplot as plt
import numpy as np
import time



def plot_pdf(pdfs):
    plt.rcParams['grid.linestyle'] = '--'
    for o in preferences.OBSERVABLES:
        plt.figure()
        edges = data_tools.get_edges(o)
        for c in preferences.CLASSES_RAW:
            plt.plot(edges, pdfs[o][c], label=constants.TRADUCTION_TABLE[c], linewidth=3)
            
            data = np.array([edges, pdfs[o][c]])
            data = data.T
            datafile_path = 'figures/'+ constants.TRADUCTION_TABLE[c] + '_' +\
            constants.PARAM_NAME_TABLE[o].replace('\\','').replace('$','').replace('{','').replace('}','') + ".txt"
            with open(datafile_path, 'w+') as datafile_id:
            #here you open the ascii file
                np.savetxt(datafile_id, data, fmt=['%1.5f','%1.5f'])
            
        plt.xlabel('{}({})'.format(constants.PARAM_NAME_TABLE[o], constants.PARAM_UNIT_TABLE[o]))
        plt.ylabel('p({})'.format(constants.PARAM_NAME_TABLE[o]))
        plt.xlim(constants.PLOT_PARAM_TABLE[o])
        plt.legend()
        plt.grid()
        plt.show()
        
        year, month, day, hour, minute, second = time.strftime("%Y,%m,%d,%H,%M,%S").split(',')

        figname = 'figures/'+\
        year +'_'+ month +'_'+ day +\
        '_'+ hour +'_'+  minute +'_'+ second + '.png'
        plt.savefig(figname)
        
        plt.pause(1)


if __name__ == "__main__":

    start_time = time.time()
    
    data_fnames = file_tools.get_data_fnames('../data/gender_compositions/')

    histograms1D = {}
    pdfs1D = {}
    # initialize empty histograms
    for o in preferences.OBSERVABLES:
        histograms1D[o], pdfs1D[o] = {}, {}
        for c in preferences.CLASSES_RAW:
            histograms1D[o][c] = data_tools.initialize_histogram(o)
            
    # compute histograms for each class
    for c in preferences.CLASSES_RAW:   
        for file_path in data_fnames[c]:
            data = np.load(file_path)
            data_A, data_B = data_tools.extract_individual_data(data)
            obs_data = data_tools.compute_observables(data_A, data_B)
            for o in preferences.OBSERVABLES:
                histograms1D[o][c] += data_tools.compute_histogram_1D(o, obs_data[o])
                
    for o in preferences.OBSERVABLES:
        for c in preferences.CLASSES_RAW:
            pdfs1D[o][c] = data_tools.compute_pdf(o, histograms1D[o][c])
            
    plot_pdf(pdfs1D)
            
    elapsed_time = time.time() - start_time
    print('\nTime elapsed  %2.2f sec' %elapsed_time)