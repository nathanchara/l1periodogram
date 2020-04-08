# -*- coding: utf-8 -*-

# Copyright 2020 Nathan Hara
#
# This file is part of the l1periodogram code.
#
# rvmodel is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# rvmodel is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with spell1.  If not, see <http://www.gnu.org/licenses/>.
"""
Least angle regression algorithm

Computes the least angle regression or lasso 
of data y in dictionary X based on Efron et al 2004


INPUTS:
    - y: 1D np array with N elements
    - X: 2D np array with N x M elements
    - lasso: computes the lasso solution if set to 1
    - tol: Convergence tolerance, iteration stop if
                norml2(y-model)/norml2(y) > tol
    - verbose: prints aindex of added and dropped variables
    - return_all_active_sets: if True the lar function returns
    not only the last active set but a list of all the active sets
    as the lars solution is explored.

OUTPUTS: 
    - active_set: indices of the lar/lasso solution
    with non zero coefficients
    - betas: coefficients that are non zero
    - Cs: Values of the absolute maximum correlation of 
    X columns with the residuals at each iteration
      with the residuals
    - normres_arr: ratio of the residual norm
    divided by the data norm
    - gammas: Values of lars/lasso steps at each iteration
    - model: output model 
    - Xact: matrix of active predictors
@author: Nathan Hara 25 January 2019
"""

import numpy as np
import math
#import matplotlib.pyplot as plt
#import time

#Global variable

#Values x such that |x|<=practical_zero are considered zero
practical_zero = 1e-13 


def lar(y,X,lasso=1, tol=0.001, verbose = 0, maxiter='default'):
    
    if verbose>=2:
        if lasso == 1:
            print('---------------- Begin LARS iterations----------------')
    if maxiter == 'default':
        maxiter = len(y)*10 
       #Maximum number of iteration
    #X,y = center_normalize(X,y)
    
    #Initialization
    #normy = np.linalg.norm(y,ord=2)
    it = 0  
    Xt = X.T
    #if X1 != None and X2 != None:
    #X1t = X1.T
    #X2t = X2.T
    active_set = [] #Set of indices of active predictors
    Cs = []    #Successive max absolute correlation 
    gammas = [] #Successive lars/lasso step sizes
    norml1betas = []
    betas = np.array([]) #Coefficients of active predictors
    normres_arr = []
    number_of_lasso_modification = 0
    model = y*0
    normres = np.linalg.norm(y-model,ord=2)
    drop = 0 #No coefficient to drop at first iteration
    keep_iterating = 1 #Condition to keep iterations
    gamma = 1
    
    
    active_set_history = [[]]
    betas_history = [[]]
    
    #Find vector most correlated with the data
    Xact, C_index, maxcorr, sj = find_max_corr(Xt,y)
    
    #print('init', Xact, C_index, maxcorr, sj)
    j = C_index
    C = maxcorr
    normres_arr.append(normres)
    norml1betas.append(0)
    Cs.append(C)
    normres_ref = normres
    
    while normres>tol and it<maxiter and keep_iterating == 1:
        #Count iterations
        it = it + 1
        
        #Update the active set
        if drop == 0 : #Check if there was no drop at previous iteration
            
            active_set.append(j)
            if verbose>=2:
                print('Variable',j+1,' omega ',np.mod(j+1,14466),'added')
            
            betas = np.append(betas,0)
            
            if it == 1:
                #sj = sj
                Xact = Xact*sj   
                #plt.figure()
                #plt.plot(Xact)
            else:
                Xact = np.c_[Xact, sj*X[:,j]]

        #Update the direction u of the LAR
        ua, Aa, wa = compute_lar_direction(Xact)    
        #if it==1: 
            #plt.plot(Xact)
        
        #Compute the signs of correlation
        Xact_notsigned = X[:,active_set]
        ca = (Xact_notsigned.T).dot(y-model)
        sa = ca/np.absolute(ca)
        d = wa*sa

        #Compute the step size in direction u
        gamma,j,sj = compute_step_size(Xt,y,model,ua,C,Aa,active_set)         
        #print('gamma,j,sj', gamma,j,sj)
        
        #v = sj*X[:,j]
        #print('Xact_notsigned.shape', Xact_notsigned.shape, 'sj*X[:,j]', v.shape)
        #print('__________________________')
        #print(gamma, j, sj)
        #print(gammabis,jbis,theta,sjbis)
        #print('__________________________')
        #----------------LASSO MODIFICATION----------------
        if lasso == 1:
            
            #Compute gammatilde i.e. minimum step so that a coefficent reaches zero
            gammatilde, index_toremove = lasso_modification(betas, d)
            compl_list = difflist(list(range(len(d))), [index_toremove])       
            drop = 0

            if gammatilde < gamma:
                
                drop = 1 #signify a variable will be dropped
                number_of_lasso_modification =\
                number_of_lasso_modification+1
                
                #Update the step length
                gamma = gammatilde #
                
                (betas, model, normres, C) = \
                    update(y,gamma, betas, ua, d, model, Xact)
                
                #Drop the appropriate variable
                if verbose >=2:
                    print('Variable',active_set[index_toremove]+1,\
                      'dropped')
                active_set.remove(active_set[index_toremove])
                Xact = Xact[:,compl_list]
                betas = np.delete(betas,index_toremove)    
                
                
            else:
                
                (betas, model, normres, C) = \
                    update(y,gamma, betas, ua, d, model, Xact)

            #!! Check the norm of the residuals has decreased
            if normres > normres_ref:
                print('---- The norm of the residuals has increased, stop iterations ----')
                keep_iterating = 0
            else:
                normres_ref  = normres
          
        #--------------------------------------------------    
        
        if lasso !=1:
        
            #Update coefficients, model, max correlation
            (betas, model, normres, C) = \
            update(y,gamma, betas, ua, d, model, Xact)
            
        gammas.append(gamma) 
        normres_arr.append(normres)
        norml1betas.append(np.sum(np.abs(betas)))
        Cs.append(C)
        
        active_set_history.append(np.array(active_set))
        betas_history.append(betas)
#            print('----')
#            print('len', len(betas), len(active_set))
#            print('len', len(betas_history[-1]), len(active_set_history[-1]))
#            print('----')
        #print('gamma',gamma)
    
    #betas2 = compute_betas(Xact_notsigned,model)
    if verbose >=1:
        print('Number of lasso modifications in lars: ',number_of_lasso_modification,
          'in', it, 'iterations')
    if it == maxiter:
        print('Maximum number of iterations reached in lars, exit')
   

    varnames = ['active_set', 'betas', 'Cs','it', 'normres_arr',
           'gammas', 'model', 'Xact','norml1betas',
           'active_set_history', 'betas_history']
    
    variables = [active_set, betas, Cs,it, normres_arr,
           gammas, model, Xact,norml1betas,
           active_set_history, betas_history]
    
    dict_out = dict(zip(varnames,variables))
    
    return(dict_out)
    
        
        
def compute_lar_direction(Xact):
    Xactt = Xact.T
    G = np.dot(Xactt,Xact)
    sh = Xact.shape
    lens = len(sh)
    if lens>1:
        one = np.ones(sh[1])
        L = np.linalg.cholesky(G)
        invG_one = np.linalg.solve(L,one)
        invG_one = np.linalg.solve(L.T,invG_one)
        #invG_one = np.linalg.solve(G,one)
       # print(np.linalg.cond(G))
    else:
        one = 1
        invG_one = 1/G    
        #print('G = ', G)
    fact = np.dot(one,invG_one)
    if fact > 0 :
        Aa = 1 / np.sqrt(fact)  
    else:
        print('LARS: Aa is imaginary')
    
    wa = Aa*invG_one
    u = Xact.dot(wa)
    u = np.multiply(1/np.linalg.norm(u,2), u)
    return(u, Aa, wa)
    
    
def find_max_corr(Xt,vect):
    
    c = Xt.dot(vect)
    c_abs = np.absolute(c)
    C_index = np.argmax(c_abs)
    C = c[C_index]
    absC = np.absolute(C)
    s = C/absC
    vect_max_correlated = Xt[C_index,:]
    vect_max_correlated.transpose()
    
    return(vect_max_correlated, C_index, absC, s)
    

def compute_step_size(Xt,y,model,u,C,Aa,active_set):
    
    #start_time = time.clock()
    global practical_zero  
    residual = y - model
    c = Xt.dot(residual)
    a = Xt.dot(u) 
    Aama = Aa - a
    Aapa = Aa + a
    Cmc = C - c
    Cpc = C + c
    nc = len(c)

    Amanz = (Aama!=0)
    Apanz = (Aapa!=0)
    gammas = np.zeros((2,nc)) #-np.ones((2,nc))
    gammas[0, Amanz] = Cmc[Amanz]/Aama[Amanz]
    gammas[1, Apanz] = Cpc[Apanz]/Aapa[Apanz]  
    gammas[gammas<=practical_zero] = math.inf
    gam_min = np.min(gammas, axis=0)
    gam_min[active_set] = math.inf
    gam_min 
    j = np.argmin(gam_min)#[gam_min>practical_zero]
    gamma = gam_min[j]

    sj = c[j]/np.absolute(c[j])
    
    return(gamma, j, sj)
    
    
    
    
def difflist(first, second):
    second = set(second)
    return [item for item in first if item not in second]

def compute_betas(Xact,model):
    Xt = Xact.T
    XtX = np.dot(Xt,Xact)
    betas = np.linalg.solve(XtX, Xt.dot(model))
    return(betas)
    
    
def lasso_modification(betas, d):

    global practical_zero
    #print('betas.shape, d.shape',betas.shape, d.shape)
    gammajs =    - betas/d
    gammatilde = math.inf 
    index_toremove = 'no_index_to_remove'
    for i in range(len(betas)):
        gammajsi = gammajs[i]
        if (gammajsi<gammatilde and gammajsi>practical_zero):
            index_toremove = i
            gammatilde = gammajsi
    
    return  gammatilde, index_toremove

        
    
def update(y, gamma, betas, ua, d, model, Xact):
    
    #Update coefficients
    betas = betas + np.multiply(gamma,d)
    #Update model
    model = model + np.multiply(gamma,ua)
    normres= np.linalg.norm(y-model,ord=2)       
    #Update max correlation
    #Do not use C = C - gamma*Aa
    cacheck = (Xact.T).dot(y-model)
    C = np.max(cacheck) 
    
    return(betas, model, normres, C)
    
    
    
    
