#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 09:40:53 2018

@author: zeynep
"""

GT_FPATH = '../data/annotations/'

SOCIAL_RELATION_FPATH = '../data/classes/'

CLASSES = ['doryo', 'kazokua', 'yujin', 'koibito']

from math import pi
import numpy as np

D_BIN_SIZE = 25 # previously 50
D_MIN_TOLERABLE = 0
D_MAX_TOLERABLE = 2000# previously 2500

D_MIN_PLOT = 0
D_MAX_PLOT = 2000

VEL_BIN_SIZE = 3/120 # previously 3/60 
VEL_MIN_TOLERABLE = 0 # m/s # prevoously -3
VEL_MAX_TOLERABLE = 2 # m/s # previously 3

ACC_BIN_SIZE = 4/200 # 
ACC_MIN_TOLERABLE = -2 #
ACC_MAX_TOLERABLE = 2 #

VEL_MIN_PLOT = 0
VEL_MAX_PLOT = 2.5

ACC_MIN_PLOT = -1
ACC_MAX_PLOT = 1

VELDIFF_BIN_SIZE = 0.5 / 60 
VELDIFF_MIN_TOLERABLE = 0 # m/s
VELDIFF_MAX_TOLERABLE = 0.5 # m/s

VELDIFF_MIN_PLOT = 0
VELDIFF_MAX_PLOT = 0.5

VDDOT_BIN_SIZE = 2 * pi / 60 
VDDOT_MIN_TOLERABLE = -pi # radians
VDDOT_MAX_TOLERABLE = pi# radians

VDDOT_MIN_PLOT = -pi
VDDOT_MAX_PLOT = pi

VVDOT_BIN_SIZE = 2 * pi / 120 
VVDOT_MIN_TOLERABLE = -pi # radians
VVDOT_MAX_TOLERABLE = pi # radians

VVDOT_MIN_PLOT = -pi
VVDOT_MAX_PLOT = pi

HEIGHT_BIN_SIZE = 7.5 # cm 
HEIGHT_MIN_TOLERABLE = 500 # mm
HEIGHT_MAX_TOLERABLE = 2105 # mm

HEIGHT_MIN_PLOT = 1250
HEIGHT_MAX_PLOT = 2000

# set the range of heightdiff to an integer multiple of HEIGHTDIFF_BIN_SIZE to 
# avoid problems in plotting pdf 
HEIGHTDIFF_BIN_SIZE = 7.5 # cm
HEIGHTDIFF_MIN_TOLERABLE = 0 # mm
HEIGHTDIFF_MAX_TOLERABLE = 1125 # mm # previously 2100

HEIGHTDIFF_MIN_PLOT = 0
HEIGHTDIFF_MAX_PLOT = 600

VELOCITY_THRESHOLD = 0.5 # m/s
DISTANCE_THRESHOLD = 2000 # mm

X_POSITION_THRESHOLD = (-10000, 50000)
Y_POSITION_THRESHOLD = (-25000, 10000)

HISTOG_PARAM_TABLE = {
    'd': (D_MIN_TOLERABLE, D_MAX_TOLERABLE, D_BIN_SIZE),
    'dx': (D_MIN_TOLERABLE, D_MAX_TOLERABLE, D_BIN_SIZE),
    'dy': (D_MIN_TOLERABLE, D_MAX_TOLERABLE, D_BIN_SIZE),
    'v_g': (VEL_MIN_TOLERABLE, VEL_MAX_TOLERABLE, VEL_BIN_SIZE),
    'a_g': (ACC_MIN_TOLERABLE, ACC_MAX_TOLERABLE, ACC_BIN_SIZE),
    'v_diff': (VELDIFF_MIN_TOLERABLE, VELDIFF_MAX_TOLERABLE, VELDIFF_BIN_SIZE),
    'vv_dot': (VVDOT_MIN_TOLERABLE, VVDOT_MAX_TOLERABLE, VVDOT_BIN_SIZE),
    'vd_dot': (VDDOT_MIN_TOLERABLE, VDDOT_MAX_TOLERABLE, VDDOT_BIN_SIZE),
    'h_avg': (HEIGHT_MIN_TOLERABLE, HEIGHT_MAX_TOLERABLE, HEIGHT_BIN_SIZE),
    'h_diff': (HEIGHTDIFF_MIN_TOLERABLE, HEIGHTDIFF_MAX_TOLERABLE, HEIGHTDIFF_BIN_SIZE),
    'h_short': (HEIGHT_MIN_TOLERABLE, HEIGHT_MAX_TOLERABLE, HEIGHT_BIN_SIZE),
    'h_tall': (HEIGHT_MIN_TOLERABLE, HEIGHT_MAX_TOLERABLE, HEIGHT_BIN_SIZE)
}

PLOT_PARAM_TABLE = {
    'd': (D_MIN_PLOT, D_MAX_PLOT),
    'dx': (D_MIN_PLOT, D_MAX_PLOT),
    'dy': (D_MIN_PLOT, D_MAX_PLOT),
    'v_g': (VEL_MIN_PLOT, VEL_MAX_PLOT),
    'a_g': (ACC_MIN_PLOT, ACC_MAX_PLOT),
    'v_diff': (VELDIFF_MIN_PLOT, VELDIFF_MAX_PLOT),
    'vv_dot': (VVDOT_MIN_PLOT, VVDOT_MAX_PLOT),
    'vd_dot': (VDDOT_MIN_PLOT, VDDOT_MAX_PLOT),
    'h_avg': (HEIGHT_MIN_PLOT, HEIGHT_MAX_PLOT),
    'h_diff': (HEIGHTDIFF_MIN_PLOT, HEIGHTDIFF_MAX_PLOT),
    'h_short': (HEIGHT_MIN_PLOT, HEIGHT_MAX_PLOT),
    'h_tall': (HEIGHT_MIN_PLOT, HEIGHT_MAX_PLOT)
}

PARAM_NAME_TABLE = {
    'v_g': r'$v_g$',
    'a_g': r'$a_g$',
    'v_diff': r'$\omega$',
    'vv_dot': r'$\theta$',
    'vd_dot': r'$\phi$',
    'd': r'$\delta$',
    'dx': r'$\delta_x$',
    'dy': r'$\delta_y$',
    'h_avg':  r'$\bar{\eta}$',
    'h_diff': r'$\Delta_{\eta}$',
    'h_short': r'${\eta}_s$',
    'h_tall': r'${\eta}_t$'
}

PARAM_UNIT_TABLE = {
    'v_g': 'm/sec',
    'a_g': 'm/sec/sec',
    'v_diff': 'm/sec',
    'vv_dot': 'rad',
    'vd_dot': 'rad',
    'd': 'mm',
    'dx': 'mm',
    'dy': 'mm',
    'h_avg': 'mm',
    'h_diff': 'mm',
    'h_short': 'mm',
    'h_tall': 'mm'
}

TRADUCTION_TABLE = {
    'koibito': 'M',
    'doryo': 'C',
    'yujin': 'Fr',
    'kazoku': 'Fa',
    'kazokuc': 'Fa+K',
    'kazokua': 'Fa-K',
    'males': 'M',
    'females': 'F',
    'mixed': 'X'
}