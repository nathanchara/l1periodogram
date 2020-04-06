# -*- coding: utf-8 -*-

# Copyright 2020 Nathan Hara
#
# This file is part of spell1.
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

import numpy as np


def covar_mat(T,err, sigmaW, sigmaR,
              sigma_calib,tau,Prot=-1, tol = 1e-4,**kwargs):
    '''
    Computes the covariance matrix, its inverse and the square 
    root of the inverse for an exopnential kernel 
    V(k,l) = (err_k**2 + sigma_W**2) * delta_kl 
           + sigmaR**2 * decay(|T[k] - T[l]|)
           
    where delta_kl is the Kronecker symbol (=1 if k=l 0 otherwise)
    decay(|T[k] - T[l]|) is set by the kwargs argument 
    (see covar_exp function)
    
    Inputs:
    - T : array of times
    - tau : see formula above
    - sigmaW : see formula above
    - sigmaR : see formula above
    - tol : matrix elements whose absolute values 
            are below tol are set to zero
    - Prot: if Prot > 0, the covariance kernel is 
             taken as quasiperiodic
    Outputs:
    - V : covariance matrix with exponential kernel (see formula above)
    
    '''   
    Id_matrix = np.eye(len(err))
    
    Vr = covar_exp(T, tau, sigmaR,Prot=Prot, tol = tol,**kwargs)
    Vw = Id_matrix*sigmaW**2
    Vcalib = covar_calib(T,sigma_calib)
    V = np.diag(err*err) + Vr + Vw + Vcalib    
  
    return(V)




def covar_exp(T, tau, sigmaR, Prot = -1, tol = 1e-8, 
              kernel='gaussian',llambda = 0.5,**kwargs):
    '''
    Computes the covariance matrix, its inverse and the square 
    root of the inverse for an exopnential kernel 
    V(k,l) = sigmaR**2 * decay(|T[k] - T[l]|)
    
    Inputs:
    - T : array of times
    - tau : see formula above
    - sigmaR : see formula above
    - tol : matrix elements whose absolute values 
            are below tol are set to zero
    - Prot: if Prot > 0, the covariance kernel is 
             taken as quasiperiodic
    - kernel: if set to 'exponential', decay(deltat) = exp(|T[k] - T[l]|/tau)
              if set to 'gaussian', decay(deltat) = exp((T[k] - T[l])**2/(2*tau**2))
    Outputs:
    - V : covariance matrix with exponential kernel (see formula above)
    - W : inverse of V
    - sqrtW : square root of W (which is a symmetric matrix)
    
    '''
    
    #llambda2 = llambda * llambda
    tau2 = tau*tau
    Nt = len(T)
    sigmaR2 = sigmaR * sigmaR 
    if tau >0:
        
        V = np.eye(Nt)*0.5
        
        for i in range(Nt):
            ti = T[i]
            for j in range(i):
                tj = T[j]
                deltaT = np.abs(ti-tj)
                if kernel =='gaussian':
                    deltaT2 = deltaT**2
                    correlationterm = 0.5*deltaT2/tau2
                elif kernel == 'exponential':
                    correlationterm = deltaT/tau
            
                    
                quasiperiodic_term = 1
                if Prot >0:
                    #sinus = np.sin(np.pi*deltaT/Prot)
                    #quasiperiodic_term = np.exp(-2/llambda2 * sinus**2)
                    quasiperiodic_term = 0.5*(1+np.cos(2*np.pi*deltaT/Prot))
                    
                V[i,j] = np.exp(-correlationterm)*quasiperiodic_term
                
            
        V = sigmaR2 * V
        
        V[np.abs(V)<tol] = 0   
        V = V + V.T 
        #for i in range(Nt):
        #    V[i,i] /= 2
    else:
        V = np.eye(Nt) * sigmaR2
    
    #Create weight matrix
    #W,sqrtW = invert_covmatrix(V,tol)
    
    return(V)
    
def covar_calib(T, sigma_calib, inst_num = None):
    # T in days !!!
    N = len(T)
    C = np.zeros((N,N))
    thres_d = 0.5
    if inst_num is None:
        inst_num = np.zeros(N)
    
    for i in range(N):
        ti = T[i]
        C[i,i] = 0.5
        insti = inst_num[i]
        for j in range(i):
            tj = T[j]
            instj = inst_num[j]
            if np.abs(ti-tj)<thres_d and insti == instj:
                C[i,j] = 1
    
                
    C = (C+C.T) * sigma_calib**2
    
    #W,sqrtW = invert_covmatrix(C,1e-13)
    
    return(C)
    
    
    
    
    
    
    
    
def covar_mats(T,err, sigmaWs, sigmaRs,sigma_calibs,taus
               ,Prot=-1, tol = 1e-4,**kwargs):
   
    nmats = len(sigmaWs)
    Vs = []
    for i in range(nmats):
        Vs.append(covar_mat(T,err, sigmaWs[i], 
                            sigmaRs[i],sigma_calibs[i],
                            taus[i],Prot=Prot, tol = tol,**kwargs))
  
    return(Vs)
    
    
def covar_mats_period(T,err, sigmaWs, sigmaRs,sigma_calibs,  
                      taus,sigmaqps, Pqps, expdecays,
                      tol = 1e-4,**kwargs):
   
    nmats = len(sigmaWs)
    Vs = []
    for i in range(nmats):
        Vrot = covar_mat(T,T*0, 0, 
                            sigmaqps[i],0,
                            expdecays[i],Prot=Pqps[i], tol = tol,**kwargs)
        
        Vred = covar_mat(T,err, sigmaWs[i], 
                            sigmaRs[i],sigma_calibs[i],
                            taus[i],Prot=-1, tol = tol,**kwargs)
        
        Vs.append(Vrot + Vred)
  
    return(Vs)   
    
    
def invert_covmatrix(V,tol=1e-13):
    
    s, vh = np.linalg.eigh(V)
    err = np.sqrt(s)
    
    if np.min(s) >1e-13: 
        
        L = np.linalg.cholesky(V)
        invL = np.linalg.inv(L)
        W = invL.T.dot(invL)
        
        #W1 = np.diag(1/s).dot(vh.T)
        #W = np.dot(vh,W1)
        #W[np.abs(W)<tol] = 0
        #W = (W + np.transpose(W))/2 #Symmetrize for numerical stability
    
        #Sqrt weight matrix
        sqrtW1 = np.diag(1/err).dot(vh.T)
        sqrtW = np.dot(vh,sqrtW1)#np.dot(aux,vh) #Weight matrix for periodogram
        sqrtW[np.abs(sqrtW)<tol] = 0
        sqrtW = (sqrtW + sqrtW.T)/2    
        
    else:
        print('The covariance matrix is not positive definite')
        W = None
        sqrtW = None
        
    return(W,sqrtW)
    