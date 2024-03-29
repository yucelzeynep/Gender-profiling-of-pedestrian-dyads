#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 09:13:15 2018

@author: zeynep
"""

import numpy as np
import math

from importlib import reload
import constants
reload(constants)
import preferences
reload(preferences)

def rotate_vectors(dxAB, dxBA, dyAB, dyBA, vxG, vyG):
    """
    Rotate to vectors to obtain an output vector whose x component is aligned with the group velocity
    """
    dAB = (dxAB**2 + dyAB**2)**.5
    dx_rAB, dy_rAB, dx_rBA, dy_rBA = dxAB, dyAB, dxBA, dyBA

    for j in range(len(dx_rAB)):
        magnitude = (vxG[j]*vxG[j] + vyG[j]*vyG[j])**.5
        if magnitude != 0:
            dx_rAB[j] = (vxG[j]*dxAB[j] +  vyG[j]*dyAB[j]) / magnitude
            dy_rAB[j] = (vyG[j]*dxAB[j] + -vxG[j]*dyAB[j]) / magnitude
 
            dx_rBA[j] = (vxG[j]*dxBA[j] +  vyG[j]*dyBA[j]) / magnitude
            dy_rBA[j] = (vyG[j]*dxBA[j] + -vxG[j]*dyBA[j]) / magnitude

    dABx = np.concatenate((dx_rAB, dx_rBA))#[dx_rAB, dx_rBA]
    dABy = np.concatenate((dy_rAB, dy_rBA))# [dy_rAB, dy_rBA]
    abs_dABy = np.abs(dABy)

    return [dAB, dABx, dABy, abs_dABy]

def extract_individual_data(data):
    """
    Separate data into different array for each individual
    """
    ids = set(data[:,1])
    id_A = min(ids)
    id_B = max(ids)
    dataA = data[data[:,1] == id_A, :]
    dataB = data[data[:,1] == id_B, :]
    # print(np.shape(dataA))
    return dataA, dataB

def compute_observables(dataA, dataB):
    """
    Compute the parameters that are used in the Bayesian inference
    """
    dxA, dyA = dataA[:, 2], dataA[:, 3]
    dxB, dyB = dataB[:, 2], dataB[:, 3]
    dxAB, dyAB = dxB - dxA, dyB - dyA
    dxBA, dyBA = -dxAB,  -dyAB
    vxA, vyA = dataA[:, 5], dataA[:, 6]
    vxB, vyB = dataB[:, 5], dataB[:, 6]
    # group velocity (linear)
    vxG, vyG = (vxA + vxB) / 2, (vyA + vyB) / 2
    vG = (vxG**2 + vyG**2)**.5
    # group acceleration (linear), to keep the array size the same repeatthe last element (acceleration is 0)
    aG = np.diff(np.hstack([vG, vG[-1]]))
    # velocities difference
    vxDiff = (vxA - vxB) / 2
    vyDiff = (vyA - vyB) / 2
    vDiff = (vxDiff**2 + vyDiff**2)**.5
    # velocities dot product
    vvdotAB = np.arctan2(vyA, vxA) - np.arctan2(vyB, vxB)
    vvdot = vvdotAB % (2 * math.pi)
    vvdot[vvdot > math.pi] = vvdot[vvdot > math.pi] - 2 * math.pi # interval is [-pi, pi]
    vvdot[vvdot < -math.pi] = vvdot[vvdot < -math.pi] + 2 * math.pi
    # velocities/distance dot product
    vddotA = np.arctan2(dyAB, dxAB) - np.arctan2(vyA, vxA)
    vddotB = np.arctan2(dyBA, dxBA) - np.arctan2(vyB, vxB)
    vddot = np.concatenate((vddotA, vddotB), axis=0)
    vddot = vddot % (2 * math.pi)
    vddot[vddot >= math.pi] = vddot[vddot >= math.pi] - 2 * math.pi # interval is [-pi, pi]
    vddot[vddot < -math.pi] = vddot[vddot < -math.pi] + 2 * math.pi
    
    # limiting vd_dot is complex because when people are betting away from each other
    # (moving in oppoiste direction) then both dot products are -
    # simiarly for heading towards each other
    # so i just leave it
    
    # heights
    hA, hB = dataA[:, 4], dataB[:, 4]
    hAvg = (hA + hB) / 2
    hDiff = np.abs(hA - hB)
    if np.mean(hA) < np.mean(hB):
        h_short = hA
        h_tall = hB
    else:
        h_short = hB
        h_tall = hA
    # rotate the vectors
    [dAB, dABx, dABy, abs_dABy] = rotate_vectors(dxAB, dxBA, dyAB, dyBA, vxG, vyG)

    observable_data = {
    'd': dAB,
    'dx': dABx[dABx>=0],
    'dy': dABy[dABy>=0],
    'v_g': vG,
    'a_g': aG,
    'v_diff': vDiff,
    'vv_dot': vvdot,
    'vd_dot': vddot,
    'h_avg': hAvg,
    'h_diff': hDiff,
    'h_short': h_short,
    'h_tall': h_tall
}
    return observable_data

def shuffle_data(data):
    """
    Shuffle the data
    """
    rng_state = np.random.get_state()
    for c, d in data.items():
        np.random.set_state(rng_state)
        np.random.shuffle(d)
        data[c] = d
    return data



def threshold(data):
    """
    Apply threshold to the various parameters
    """
    dataA, dataB = extract_individual_data(data)
    # dataA, dataB = threshold_position(dataA, dataB)
    dataA, dataB = threshold_distance(dataA, dataB)
    dataA, dataB = threshold_velocity(dataA, dataB)
    dataA, dataB = threshold_data_veldiff(dataA, dataB)
    dataA, dataB = threshold_vvdot(dataA, dataB)
    dataA, dataB = threshold_height(dataA, dataB)
    
    return np.concatenate((dataA, dataB), axis=0)

def threshold_position(dataA, dataB):
    """
    Apply a threshold on the velocity to the given data
    """
    timeA = dataA[:, 0]
    timeB = dataB[:, 0]

    xA, yA = dataA[:, 2], dataA[:, 3]
    xB, yB = dataB[:, 2], dataB[:, 3]

    threshold_xA = np.logical_and(constants.X_POSITION_THRESHOLD[0] < xA, xA < constants.X_POSITION_THRESHOLD[1])
    threshold_yA = np.logical_and(constants.Y_POSITION_THRESHOLD[0] < yA, yA < constants.Y_POSITION_THRESHOLD[1])
    thresholdA = np.logical_and(threshold_xA, threshold_yA)

    threshold_xB = np.logical_and(constants.X_POSITION_THRESHOLD[0] < xB, xB < constants.X_POSITION_THRESHOLD[1])
    threshold_yB = np.logical_and(constants.Y_POSITION_THRESHOLD[0] < yB, yB < constants.Y_POSITION_THRESHOLD[1])
    thresholdB = np.logical_and(threshold_xB, threshold_yB)

    pre_thresholdA = timeA[thresholdA]
    pre_thresholdB = timeB[thresholdB]
    inter_threshold = np.intersect1d(pre_thresholdA, pre_thresholdB)

    threshold_boolA = np.isin(timeA, inter_threshold)
    threshold_boolB = np.isin(timeB, inter_threshold)

    thresholdA = dataA[threshold_boolA, :]
    thresholdB = dataB[threshold_boolB, :]

    return thresholdA, thresholdB

def threshold_distance(dataA, dataB):
    """
    Apply a threshold on the distance to the given data
    """
    distAB = ((dataA[:, 2] - dataB[:, 2])**2 + (dataA[:, 3] - dataB[:, 3])**2)**.5
    thresholdA = dataA[distAB < constants.DISTANCE_THRESHOLD, :]
    thresholdB = dataB[distAB < constants.DISTANCE_THRESHOLD, :]

    return thresholdA, thresholdB

def threshold_velocity(dataA, dataB):
    """
    Apply a threshold on the velocity to the given data
    """
    timeA = dataA[:, 0]
    timeB = dataB[:, 0]

    vxA, vyA = dataA[:, 5], dataA[:, 6]
    vxB, vyB = dataB[:, 5], dataB[:, 6]

    vA = (vxA**2 + vyA**2)**.5
    vB = (vxB**2 + vyB**2)**.5

    pre_thresholdA = timeA[vA > constants.VELOCITY_THRESHOLD]
    pre_thresholdB = timeB[vB > constants.VELOCITY_THRESHOLD]
    inter_threshold = np.intersect1d(pre_thresholdA, pre_thresholdB)

    threshold_boolA = np.isin(timeA, inter_threshold)
    threshold_boolB = np.isin(timeB, inter_threshold)

    thresholdA = dataA[threshold_boolA, :]
    thresholdB = dataB[threshold_boolB, :]

    return thresholdA, thresholdB

def threshold_data_veldiff(dataA, dataB):
    """
    Apply a threshold on the velocity difference to the given data
    """
    vxA, vyA = dataA[:, 5], dataA[:, 6]
    vxB, vyB = dataB[:, 5], dataB[:, 6]
    veldiff_x = (vxA - vxB) / 2
    veldiff_y = (vyA - vyB) / 2
    veldiff = (veldiff_x**2 + veldiff_y**2)**.5
    threshold_bool = np.logical_and(constants.VELDIFF_MIN_TOLERABLE < veldiff, veldiff < constants.VELDIFF_MAX_TOLERABLE)
    thresholdA = dataA[threshold_bool, :]
    thresholdB = dataB[threshold_bool, :]

    vxA, vyA = thresholdA[:, 5], thresholdA[:, 6]
    vxB, vyB = thresholdB[:, 5], thresholdB[:, 6]

    veldiff_x = (vxA - vxB) / 2
    veldiff_y = (vyA - vyB) / 2
    veldiff = (veldiff_x**2 + veldiff_y**2)**.5
    return thresholdA, thresholdB

def threshold_vvdot(dataA, dataB):
    """
    Apply a threshold on the dot product of the velocities to the given data
    """
    vxA, vyA = dataA[:, 5], dataA[:, 6]
    vxB, vyB = dataB[:, 5], dataB[:, 6]

    vA = (vxA**2 + vyA**2)**.5
    vB = (vxB**2 + vyB**2)**.5
    vxDiff = (vxA - vxB) / 2
    vyDiff = (vyA - vyB) / 2
    vDiff = (vxDiff**2 + vyDiff**2)**.5

    vvdotAB = np.arctan2(vyA, vxA) - np.arctan2(vyB, vxB)
    # limit vvdot to the interval [-pi, pi]
    vvdot = vvdotAB % (2 * math.pi)
    vvdot[vvdot > math.pi] = vvdot[vvdot > math.pi] - 2 * math.pi # interval is [-pi, pi]
    vvdot[vvdot < -math.pi] = vvdot[vvdot < -math.pi] + 2 * math.pi

    threshold_bool = np.logical_and(constants.VDDOT_MIN_TOLERABLE < vvdot, vvdot < constants.VDDOT_MAX_TOLERABLE)
    thresholdA = dataA[threshold_bool, :]
    thresholdB = dataB[threshold_bool, :]
    
    vxA, vyA = thresholdA[:, 5], thresholdA[:, 6]
    vxB, vyB = thresholdB[:, 5], thresholdB[:, 6]

    vA = (vxA**2 + vyA**2)**.5
    vB = (vxB**2 + vyB**2)**.5
    vxDiff = (vxA - vxB) / 2
    vyDiff = (vyA - vyB) / 2
    vDiff = (vxDiff**2 + vyDiff**2)**.5
    return thresholdA, thresholdB

def threshold_height(dataA, dataB):
    """
    Apply a threshold on the height to the given data
    """
    hA = dataA[:, 4]
    hB = dataB[:, 4]

    condA = np.logical_and(constants.HEIGHT_MIN_TOLERABLE < hA, hA < constants.HEIGHT_MAX_TOLERABLE)
    condB = np.logical_and(constants.HEIGHT_MIN_TOLERABLE < hB, hB < constants.HEIGHT_MAX_TOLERABLE)
    cond = np.logical_and(condA, condB)

    thresholdA = dataA[cond, :]
    thresholdB = dataB[cond, :]

    return thresholdA, thresholdB


def initialize_histogram(obs):
    """
    Initialize an empty histogram for the given observable
    """
    (min_bound, max_bound, bin_size) = constants.HISTOG_PARAM_TABLE[obs]
    return np.zeros((round((max_bound - min_bound) / bin_size)))


def compute_histogram_1D(obs, obs_data):
    """
    Compute the histogram of the given observable data
    """
    if len(obs_data) == 0:
        print('problem')
    
    (min_bound, max_bound, bin_size) = constants.HISTOG_PARAM_TABLE[obs]
    n_bins = round((max_bound - min_bound) / bin_size) + 1
    edges = np.linspace(min_bound, max_bound, n_bins)
    
   
    # since we remove thresholding, values outside the borders of histogram limits
    # may appear in computation of histogram.
    # but we do not want them in training pdfs, so below I remove them from the 
    # list of observed values
    obs_data = [j for j in obs_data if j >= min_bound]
    obs_data = [j for j in obs_data if j <= max_bound]
    
    histog = np.histogram(obs_data, edges)
    
    return histog[0]



def compute_pdf(obs, histogram):
    """
    Compute the PDF of the given observable histogram
    """
    (_, _, bin_size) = constants.HISTOG_PARAM_TABLE[obs]
    pdf = histogram / sum(histogram) / bin_size
    return pdf


def get_edges(obs):
    """
    Compute the abscissa value to plot the PDF of the given observable parameter
    """
    (min_bound, max_bound, bin_size) = constants.HISTOG_PARAM_TABLE[obs]
    return np.arange(min_bound, max_bound, bin_size)

def find_bins(obs, obs_data):
    """
    Find the bins that corresponds to each value in obs_data
    """
    (min_bound, max_bound, bin_size) = constants.HISTOG_PARAM_TABLE[obs]
    n_bins = round((max_bound - min_bound) / bin_size) + 1
    edges = np.linspace(min_bound, max_bound, n_bins)
    bins = np.digitize(obs_data, edges)
    
    # for some reason, digitize gives a value between 1 and n_bins +1,
    # as if it is using matlab/fortran logic. 
    # but i use it like in C logic in the probability update
    bins[:] = [x - 1 for x in bins]    
    
    return bins
