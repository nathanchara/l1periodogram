#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 16:13:04 2019

@author: nathan
"""
import numpy as np


def filterpoly(T,y,W2,M0,n):
    
    
    Tnorm = T/np.linalg.norm(T)
    Nt = len(T)
    vnull = T*0
    
    #if M0 != 0:
    s = M0.shape
    l0 = s[1]
    #else:
    #    l0 = 1
    
    M = np.zeros((Nt, n+l0))
    #Offsets
    #if M0 != 0:
    s = M0.shape
    l0 = s[1]
    n_dropped_columns = 0
    for i in range(l0):
        i = i - n_dropped_columns
        v = M0[:,i]
        u = vnull
        for j in range(i):
            u = u + np.dot(M[:,j], v)*M[:,j]   
        v = v - u
        normv = np.linalg.norm(v)
        #if normv> 1e-13:
        M[:,i] = v/normv
        #else:
        #    print(M[0:3,:])
        #    M = np.concatenate((M[:,0:i], M[:,i+1:-1]),axis=1)
        #    print(M[0:3,:])
        #    n_dropped_columns = n_dropped_columns + 1
    #print(M[0:3,:])  
    #else: 
    #    v = vnull + 1
    #    M[:,0] = v/np.linalg.norm(v)
    
    #Polynomials
    newl0 = l0 - n_dropped_columns
    inds = range(newl0,newl0+n) 
    for i in inds:
        if i==l0:
            v = Tnorm
        else:
            v = M[:,i-1]*Tnorm
        for j in range(i):
            v = v - np.dot(M[:,j], v)*M[:,j]           
        '''
        u = vnull
        for j in range(i):
            u = u + np.dot(M[:,j], v)*M[:,j]
        v = v - u
        '''
        
        M[:,i] = v/np.linalg.norm(v)
    #for i in range(n+1, n+2+l0):
    #    v = M0[]
    
    MtW2 = np.dot(M.T,W2)
    G = np.dot(MtW2,M)
    b = np.dot(MtW2,y)
    theta = np.linalg.solve(G,b)
    
    model = np.dot(M,theta)
    #print(np.linalg.cond(M))
    return(model, theta, M)
    