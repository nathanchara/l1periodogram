#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:22:58 2020

@author: nathan
"""


#import objgraph
import numpy as np
import matplotlib.pyplot as plt

import l1periodogram_v1
import covariance_matrices
import combine_timeseries
import filter_poly


# ------------ Load the data ------------ #
#'HD158259'#
filepath = 'data/'
#dataset_names = [filepath+target_name+'.dat']
dataset_names = ['HD69830_2006.dat']#'HD10180_2011.dat'HD69830_2006
sigmas_inst = [0]

#dataset_names = ['55 Cnc_ELODIE_DRS-TACOS.rdb', \
#                 '55 Cnc_HAMILTON_Pub-2008.rdb',\
#                 '55 Cnc_HARPN_DRS-3-7.rdb', \
#                 '55 Cnc_HIRES_Pub-2017.rdb', \
#                 '55 Cnc_HRS_Pub-2004.rdb']

#dataset_names = ['55 Cnc_HIRES_Pub-2017.rdb', \
#                 '55 Cnc_HRS_Pub-2004.rdb']
#sigmas_inst = [0., 0., 0., 0., 0.]

#dataset_names = ['GJ 876_HAMILTON_Pub-2001.rdb', \
#                 'GJ 876_HARPS03_DRS-3-5.rdb',\
#                 'GJ 876_HIRES_Pub-2017.rdb']
#sigmas_inst = [0., 0., 0.]
dataset_names = [filepath+d for i,d in enumerate(dataset_names)]
target_name = dataset_names[0][:7]


outlier_mad_cond = 3.5
T,y,err,offsets,dico, dict_out, dataset_names_out = \
                        combine_timeseries.create_dataset(dataset_names,
                        outlier_cond = outlier_mad_cond,
                        sigmas_inst=sigmas_inst, bintimescale = 0.7)
                        
# ------------ Covariance model ------------ #
sigmaW = 1.#[1.0]
sigma_calib = 0.
sigmaR = 0.
tau = 0.
#[6.0]#[0.0]# #Correlation timescale
#sigmaqp = [2.0]#[3.0]#[0.]#
#periodqp = [11.6]#[34.0]#[1.]#
#decaytime = [10.0]#[60.0]#[1.]#



#Define the class
c = l1periodogram_v1.l1p_class(T,y)
Nt = len(T)


c.dataset_names = dataset_names_out
c.offsets = offsets

#Model
V = covariance_matrices.covar_mat(T,err, sigmaW, sigmaR,
              sigma_calib,tau,Prot=-1, tol = 1e-8,kernel='gaussian')

c.set_dict(omegamax = 3*np.pi, 
                 V = V,
                 MH0 = offsets,
                 Nphi=10,
                 numerical_method='lars')
#Plot the input data of the l1 periodogram

c.plot_input_data()

#c.unpenalize_periods([14.651808, 5154, 44.390595, 0.736544686, 261.32])

#Run l1 periodogram
c.l1_perio(significance_evaluation_methods=['fap', 'evidence_laplace'],
max_n_significance_tests = 9,starname = target_name,
       smoothing='fit', plot_output=False)

#reweight_args = {'Nit':0, 'q':0, 'eps':1e-10},

print('Peaks at ', 2*np.pi/c.omega_peaks[:12], ' days')
print('With amplitude ', c.peakvalues)
print('With log10fap ', c.significance['log10faps'])
print('With approx. log10 evidence  ', c.significance['log10_bayesf_laplace'])


c.plot_with_list(9)
mmm

#c.smooth_solution(smoothing='fit')
#c.evaluate_significance(15)
#c.find_peaks()
#c.plot_clean(15, title='55 Cnc, l1 periodogram', save=True)

c.plot_clean(10,annotations='log10_bayesf_laplace')

c.l1_perio(significance_evaluation_methods=['fap', 'evidence_laplace'],
max_n_significance_tests = 9,starname = target_name,
       smoothing='fit')


c.unpenalize_periods([61.05423, 30.2198, 15.04, 1.93787047, 60.6432748, 30.1692673],offsets, numerical_method='lars')
#c.unpenalize_periods([8.6640])
c.l1_perio(max_n_significance_tests = 14,smoothing='fit')




c.update_dict(omegamax = 1.9*np.pi)
c.l1_perio(max_n_significance_tests = 9,smoothing='fit')



a,b,M1 = filter_poly.filterpoly(c.t,c.y,c.W2,offsets,1)
c.update_dict(MH0 = M1)
c.l1_perio(max_n_significance_tests = 9,smoothing='fit')

c.plot_with_list(14,annotations='log10faps')

c.set_dict(omegamax = 1.9*np.pi, 
                 V = V,
                 MH0 = offsets,
                 Nphi=10,
                 numerical_method='gglasso')
c.l1_perio(significance_evaluation_methods=['fap', 'evidence_laplace'],
max_n_significance_tests = 9,starname = target_name,
       smoothing='fit')




c.set_dict(numerical_method='lars')
c.unpenalize_periods([61.05423, 30.2198],offsets, numerical_method='lars')
#c.unpenalize_periods([8.6640])
c.l1_perio(max_n_significance_tests = 14,smoothing='fit')
