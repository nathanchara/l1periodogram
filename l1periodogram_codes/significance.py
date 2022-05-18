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
# along with l1periodogram.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import fastlinsquare_cholesky
import math as ma
from scipy.optimize import minimize

# ------------- Minimizations ------------- #

def residuals(Tm, y, W, omegas_planets, Mat,
              return_matrix= False, return_only_chi2 = False):
   """
   Computes the residuals after fitting sine functions at frequencies omegas_planets to data y
   Inputs:
   - Tm: times of measurement
   - y: raw time series 
   - err: measurement errors
   - omegas_planets: frequencies of the planets detected so far (frequency defined as 2*pi/period)
     As an np array
   Outputs
   - residuals_timeseries: time series with the linear model 
        [cos(omega_1) ... cos(omega_n) sin(omega_1) ... sin(omega_n)  1]
     is  fitted
   - x: fitted paramameters
   """
   #print(ncol)
   matrix = create_om_mat(Tm,Mat,omegas_planets)
   #fit 
#   AtW = np.dot(matrix.T, W)
#   A = np.dot(AtW, matrix)
#   b = np.dot(AtW, y)
#   x = np.linalg.solve(A,b) #fitted parameters
   chi2,x,Covariance = \
          fastlinsquare_cholesky.fastchi2lin(matrix,W,y, 
                                    compute_covar=False,
                                    full_chi2 = True)
   model_out = np.dot(matrix,x)
   #print(A.shape, x.shape)

   #Substract the model from the data
   residuals_timeseries = y - model_out
    
   if return_only_chi2:
       return(chi2)
       
   elif return_matrix:
       return(residuals_timeseries, x, matrix)
   else:
       return(residuals_timeseries, x)
   
    
    
def residuals_nl( T, y, W2, Mat, omegas_planets,
                 **kwargs):
    
    result = minimize(lambda x:chi2_omegas(T, y, W2, Mat,x),
                   omegas_planets,
                   #jac = lambda x:jacob(T, y, W2, Mat,x),
                   **kwargs)
    chi2 = y.T.dot(W2).dot(y) + result.fun
    xout = result.x
    
    success_minim = result.success
    
    if success_minim != True:
        result = minimize(lambda x:chi2_omegas(T, y, W2, Mat,x),
                       omegas_planets,
                       #jac = lambda x:jacob(T, y, W2, Mat,x),
                       method='TNC')
        chi2 = y.T.dot(W2).dot(y) + result.fun
        xout = result.x
    success_minim = result.success
    
    if success_minim != True:
        result = minimize(lambda x:chi2_omegas(T, y, W2, Mat,x),
                       omegas_planets, 
                       #jac = lambda x:jacob(T, y, W2, Mat,x),
                       method='Nelder-Mead')
        chi2 = y.T.dot(W2).dot(y) + result.fun
        xout = result.x
    success_minim = result.success  
    
    if success_minim != True:
        print('The minimisation did not converge')
        chi2 = residuals( T, y, W2, omegas_planets, Mat,
              return_matrix = False, return_only_chi2 = True)
        xout = omegas_planets
        
    return(chi2, xout)
    
    
def jacob(T, y, W2, Mat,omegas):
    
    A = create_om_mat(T,Mat,omegas)
    nomegas = len(omegas)
    jacobian = np.zeros(nomegas)
    G = A.T.dot(W2).dot(A)
    b = A.T.dot(W2).dot(y)
    
    invGb = np.linalg.solve(G,b)
    btinvG = invGb.T
    
    for i in range(nomegas):
        om = omegas[i]
        Aprime = A.copy()
        Aprime[:,i] = -om*np.sin(om*T)
        Aprime[:,nomegas+i] = om*np.cos(om*T)
        AprimetW2 = Aprime.T.dot(W2)
        
        Gprime1 = Aprime.T.dot(W2).dot(A) 
        Gprime = Gprime1.T +  Gprime1
        
        term1 = -2*btinvG.dot(AprimetW2).dot(y)
        term2 = btinvG.dot(Gprime).dot(invGb)
        
        #print(term1, term2)
        jacobian[i] = term1 + term2
    
    #print(np.linalg.norm(jacobian))
                         
    return(jacobian)
     
    
def chi2_omegas(T, y, W2, Mat, x):
    
    A  = create_om_mat(T,Mat,x)
#    chi2_out = fastlinsquare_cholesky.fastchi2lin(A,W2,y)
#    chi2_trunc = -chi2_out[0]

    chi2t, a, b = fastlinsquare_cholesky.fastchi2lin(
                 A,W2,y,check_invertibility = True,
                compute_covar=False, full_chi2 = False)
    chi2_trunc = - chi2t

#    AtW =  A.T.dot(W2)
#    AtWy = AtW.dot(y)
#    AtWA = AtW.dot(A)    
#    try:
#        L = np.linalg.cholesky(AtWA)
#    except:
#        return(np.inf)
#    
#    aux =  np.linalg.solve(L, AtWy)
#    aux2 = aux.dot(aux)
#    chi2_trunc =  - aux2
    
    return(chi2_trunc)    

#def chi2_omegas(T, y, W2, Mat, x):
#    
#    A  = create_om_mat(T,Mat,x)
##    chi2_out = fastlinsquare_cholesky.fastchi2lin(A,W2,y)
##    chi2_trunc = -chi2_out[0]
#    AtW =  A.T.dot(W2)
#    AtWy = AtW.dot(y)
#    AtWA = AtW.dot(A)    
#    aux =  np.linalg.solve(AtWA, AtWy)
#    aux2 = AtWy.dot(aux)
#    chi2_trunc =  - aux2
#    
#    return(chi2_trunc)
    
def create_om_mat(T,Mat,x):

    Nt = len(T)
    nothervects = Mat.shape[1]
    nomegas = len(x)
    ncol = nomegas*2+nothervects
    matrix = np.zeros((Nt,ncol))
       
    for i in range(nomegas):
       matrix[:,i]         = np.cos(x[i]*T)
       matrix[:,i+nomegas] = np.sin(x[i]*T)
    matrix[:,nomegas*2:nomegas*2+nothervects] = Mat[:,range(nothervects)]
    
    return(matrix)

    
# ------------- False alarm probabilities ------------- #
    
def fap(Tm, err, omegamax_periodogram, max_periodogram_power, n0):
   """
   Computes the FAP of the Generalized Lomb Scargle Periodogram  (Zechmeister & Kurster 2009)
   with the Baluev 2008 formula
   Inputs:
   - Tm: epochs of measurement
   - err: measurement errors
   - omegamax_periodogram: max frequency of the periodogram (frequency defined as 2*pi/period)
   - max_periodogram_power: maximum value of the periodogram
   output
   - log10FAP: log base 10 of the false alarm probability of the maximum of the periodogram
   """
   Nt = len(Tm)
   NK = Nt - n0 - 2
   NH = Nt - n0
   w = 1/(err*err)
   w /= np.sum(w)
   meant = np.sum(Tm*w)
   meant2 = np.sum(Tm*Tm*w)
   DtFap = meant2 - meant * meant
   pmin = omegamax_periodogram/(2*np.pi)
   WFap = np.sqrt(4.0 * np.pi * DtFap) * pmin
   chi2ratio = 1.0 - max_periodogram_power

   FapSingle = chi2ratio**(NK / 2.0)
   
#   print('wfap = ', WFap)
#   print('FapSingle = ', FapSingle)
#   print('max_periodogram_power = ', max_periodogram_power)
#   print('NH = ', NH)
#   print('chi2ratio = ', chi2ratio)
#   print('-----------------')
   tauFap = WFap * FapSingle * np.sqrt(max_periodogram_power * NH / (2.0 * chi2ratio))
   if (max(tauFap, FapSingle)<1e-10):
     return(np.log10(tauFap+FapSingle))
   else:
     return(np.log10(1 - (1 - FapSingle) * np.exp(-tauFap)))   
   

def fap2( Tm, err, omegamax_periodogram, max_periodogram_power, n0):
   """
   Computes the FAP of the Generalized Lomb Scargle Periodogram  (Zechmeister & Kurster 2009) 
   with the Baluev 2008 formula
   Inputs:
   - Tm: epochs of measurement
   - err: measurement errors
   - omegamax_periodogram: max frequency of the periodogram (frequency defined as 2*pi/period)
   - max_periodogram_power: maximum value of the periodogram
   output
   - log10FAP: log base 10 of the false alarm probability of the maximum of the periodogram 
   """
   Nt = len(Tm)
   NK = Nt - n0 - 2
   NH = Nt - n0
   d = 2

   w = 1/(err*err)
   sumw = np.sum(w);
   #bar = @(x) sum(w.*x)/sumw
   barT = np.sum(w*Tm)/sumw
   barT2 = np.sum(w*Tm*Tm)/sumw

   #Factor Afmax
   Dt =  barT2 - barT*barT;
   Teff = np.sqrt(4*np.pi*Dt)
   Aomegamax = 2*np.power(np.pi,1.5)*omegamax_periodogram*Teff
   Afmax = Aomegamax/(2*np.pi)

   #Factor gamma
   #gammaf = (ma.gamma(0.5*(NH))/ma.gamma(0.5*(NK+1)))/(2*np.pi)
   try:
       log10gammaf = (np.lgamma(0.5*(NH)) - np.lgamma(0.5*(NK+1)) \
               - np.log(2*np.pi)) / np.log(10)
   except:
       log10gammaf=0

   if max_periodogram_power>1e-13:
       Z = 0.5*NH*max_periodogram_power #defined as z1 in Baluev 2008
       factor1 = 2*Z/(np.pi*NH)
       factor2 = 1-2*Z/NH

    
       log10FAP = log10gammaf + np.log10(Afmax) + 0.5*(d-1)*np.log10(factor1)\
                              + 0.5*(NK-1)*np.log10(factor2)
   else:
       log10FAP = 0
    
   return(log10FAP)
   
   
def fapV( Tm, invV, omegamax_periodogram, max_periodogram_power, n0):
    '''
    Computes the FAP for periodograms with general covariance model (non necessarily diagonal) following Delisle, Hara, Ségransan 2020
    Astronomy & Astrophysics, Volume 635, id.A83, 8 pp.
    
    Inputs:
    - Tm: time array of observations (np array)
    - invV: invert of the covariance matrix of the noise
    - omegamax_periodogram: maximum frequency of the periodogram, assumed to have
    a tightly spaced grid from 0 to omegamax_periodogram
    - max_periodogram_power: maximum of the periodogram, defined as
      Periodogram(omega) =  (chi2(model H0 and sine at omega) - chi2(model H0)) / chi2(model H0)
    '''
    
    #L = np.linalg.cholesky(V)
    #invL = np.linalg.inv(L)
    #invV = invL.T.dot(invL)
    N = len(Tm)
    
    SsincNumaxDelta = 0
    SSigmasincNumaxDelta = 0
    SPisincNumaxDelta = 0
    
    NH = N - n0
    NK = NH - 2
    
    FapSingle =  (1-2*max_periodogram_power/NH)**(NK/2)
    
    for i in range(N):
       
        ti = Tm[i]
        
        for j in range(N):
         
            tj = Tm[j]       
            deltaij = ti - tj
            Piij = ti * tj
            Sigmaij = ti + tj
            invVsincNumaxDeltaij = invV[i,j]*np.sinc(omegamax_periodogram*deltaij)
            
            SsincNumaxDelta += invVsincNumaxDeltaij
            SSigmasincNumaxDelta += invVsincNumaxDeltaij*Sigmaij
            SPisincNumaxDelta += invVsincNumaxDeltaij*Piij
        
    D = SPisincNumaxDelta/SsincNumaxDelta - (SSigmasincNumaxDelta/SsincNumaxDelta)**2/4

    Teff = np.sqrt(4*np.pi) * np.sqrt(D)
    W = omegamax_periodogram/(2*np.pi) * Teff   
  
    try:
        gammaH = np.sqrt(2/NH)*ma.gamma(NH/2) / ma.gamma((NH-1)/2)   
    except:
        gammaH = 1
    chi2ratio = 1.0 - max_periodogram_power
    FapSingle = chi2ratio**(NK / 2.0)
    
    fact =  chi2ratio**((NK-1)/2)
    sqrtZ = np.sqrt(max_periodogram_power*NH/2)
    tauFap = gammaH*W*(1-2*max_periodogram_power/NH)**(NK/2)*fact*sqrtZ
    
    
    if (max(tauFap, FapSingle)<1e-10):
        return(np.log10(tauFap+FapSingle))
    else:
        return(np.log10(1 - (1 - FapSingle) * np.exp(-tauFap)))     
   
   
   
   
def logEv_linear_params(T,y,W2,omegas, M0, Sigma = None):

    nH = M0.shape[1]
    Nom = len(omegas)
    N = len(T)
    p = nH + 2*Nom
    if Sigma is None:
        Sigma = np.eye(p)*(3*1e3)**2
    res, linparams = residuals(T, y, W2, omegas, M0,
              return_matrix= False, return_only_chi2 = False)

    A = create_om_mat(T,M0,omegas)
    invSigma = np.linalg.inv(Sigma)
    invZ = A.T.dot(W2).dot(A) + invSigma
    #Z = np.linalg.inv(invZ)
    c = y.T.dot(W2).dot(y)
    b = A.T.dot(W2).dot(y)
    d = c - b.dot(np.linalg.solve(invZ,b))
    
    LinvZ = np.linalg.cholesky(invZ)
    LW2 = np.linalg.cholesky(W2)
    LSigma = np.linalg.cholesky(Sigma)
    
    halflogdetinvZ =   np.sum(np.log(np.diag(LinvZ)))
    halflogdetV = - np.sum(np.log(np.diag(LW2)))
    halflogdetSigma= np.sum(np.log(np.diag(LSigma)))
    
    logevlin = -N*np.log(2*np.pi) - halflogdetV \
                   - halflogdetSigma \
                 - halflogdetinvZ - 0.5*d
    
    return(logevlin)

#------------- Laplace approximation of the evidence ------------ #


def logEv_omegas(T,y,W2,omegas, M0):
    
    if len(omegas)>0:
        chi2, omegasnl = residuals_nl( T, y, W2, M0, omegas)
    else:
        omegasnl = omegas
    
    res, linparams = residuals(T, y, W2, omegasnl, M0,
              return_matrix= False, return_only_chi2 = False)
    
    logEvlaplace = logEv(T,y,W2,omegasnl, linparams, M0)
    
    return(logEvlaplace)
    

def logEv(T,y,W2,omegas,linparams,  M0) :
    '''
    Compute the Laplace approximation of the evidence of a model
    consisting of a sum of sinusoids at frequencies omegas (in rad/day)
    with eq. 5 in Kass & Raftery 1995, J. American Stat. Assoc., Vol. 90, No. 430.)
    
    The model is v = M0.dot(C) + Sum_i A_i * cos(omegas[i]*T) + B_i * sin(omegas[i]*T)
    Inputs:
        - T: time arrays of the observations (in days) (N components vector)
        - y: data (time-series) (N components vector)
        - W2: inverse of the covariance matrix of the noise (NxN matrix)
        - omegas: frequencies of the mo
        - linparams: list of linear parameters A_1 ... A_d B_1 ... B_d C_1 ... C_M
        - M0: null hypothesis model (N x M matrix)
    
    Outputs:
        -
    '''
    m = model(T,omegas, linparams, M0)
    res = y - m
    Wres = W2.dot(res)
    chi2 = res.dot(Wres)
    N = len(T)
    Np = len(omegas)
    #Nt = len(T)
    Nlp = len(linparams)
    nparams = Np + Nlp
    chi2= Wres.dot(res)
    LW = np.linalg.cholesky(W2)
    halflogdetV = - np.sum(np.log(np.diag(LW)))
    
    information_matrix = - hessian(T,y,W2,omegas, linparams, M0)
    
    #Compute the determinant
    try:
        L = np.linalg.cholesky(information_matrix)
        diagL = np.diag(L)
        halflogdet_inverse_info_matrix = - np.sum(np.log(diagL))
    #print(np.diag(L))
    except:
        print('Warning: cannot compute the Cholesky factorization of the log-likelihood Hessian, computing svd')
        U, s, V = np.linalg.svd(information_matrix, full_matrices=False)
        halflogdet_inverse_info_matrix = - np.sum(np.log(s))
        #raise Exception('Cannot compute the Cholesky factorization of the log-likelihood Hessian')
        

#    if np.sum(diagL<=0) >0:
#        eigvals, eigvecs = np.linalg.eig(inverse_information_matrix)
#        print('Eigenvalues of the score Hessian', -eigvals)
#        print('Determinant of the score Hessian',np.linalg.det(inverse_information_matrix))
#        raise Exception('score Hessian is not positive definite')

    log_mle = - 0.5*N*np.log(2*np.pi) - halflogdetV - 0.5*chi2
    logev = 0.5*nparams*np.log(2*np.pi) \
            + halflogdet_inverse_info_matrix + log_mle
            
    #print('insinemodel_laplace, logev', logev)
            #0.5*np.log(np.linalg.det(W2)) \
            #0.5*np.log(np.linalg.det(-Hessian_K)) + \+  0.5*nparams_sigma*np.log(2*np.pi)
#    print(- 0.5*chi2, - 0.5*N*np.log(2*np.pi),
#          0.5*np.log(np.linalg.det(W2)),
#          - 0.5*np.log(np.linalg.det(-Hessian_K)),
#          0.5*nparams_sigma*np.log(2*np.pi))
    if np.isnan(logev):
        #print(np.linalg.det(inverse_information_matrix))
        print(halflogdet_inverse_info_matrix)
        raise Exception('log evidence is nan')
    return(logev)


def hessian(T,y,W2,omegas, linparams, M0):
    
    Np = len(omegas)
    #Nt = len(T)
    Nlp = len(linparams)
    nparams = Np+Nlp
    md,md2 = model_derivs(T,omegas, linparams, M0)
    
    hessian_term1 = 2*md.T.dot(W2).dot(md)
    hessian_term2 = np.zeros((nparams, nparams))
    
    m = model(T,omegas, linparams, M0)
    res = y - m
    Wres = W2.dot(res)
    
    for i in range(nparams):
        for j in range(nparams):
            hessian_term2[i,j] = - 2 * Wres.dot(md2[:,i,j])
            
    #print(hessian2.shape)
    #hessian2 = (hessian2 + hessian2.T) - np.diag(np.diag(hessian2))
    hessian_loglikelihood = -0.5 * (hessian_term1 + hessian_term2)
    
    return(hessian_loglikelihood)
            


def model(T,omegas, linparams, M0):
    M = create_om_mat(T,M0,omegas)
    return(M.dot(linparams))
    

def model_derivs(T,omegas, linparams, M0):
    
    Np = len(omegas)
    Nt = len(T)
    Nlp = len(linparams)
    As = linparams[:Np]
    Bs = linparams[Np:Np*2]
    nparams = Np+Nlp
    md = np.zeros((Nt, nparams))
    md2 = np.zeros((Nt, nparams,nparams))
    
    
    for i in range(Np):
        om = omegas[i]
        cosv = np.cos(om*T)
        sinv  = np.sin(om*T)
        md[:,3*i]   = cosv
        md[:,3*i+1] = sinv
        md[:,3*i+2] = T*(-As[i]*sinv +Bs[i]*cosv)
        
        md2[:,3*i+2,3*i+2] = - T**2 * (As[i]*cosv +Bs[i]*sinv)
        md2[:,3*i,3*i+2] = -T*sinv
        md2[:,3*i+2,3*i] = -T*sinv
        md2[:,3*i+1,3*i+2] = T*cosv
        md2[:,3*i+2,3*i+1] = T*cosv

    md[:,3*Np:] = M0
    
    return(md, md2)




def training_sets(N, Nsim=30, Training_prop = 0.6, method = 'random'):
    
    '''
    Define a list with Nsim elements, each of them are np arrays of size 
    int(Training_prop*N) of indices between 0 and N-1
    This defines the training set indices for cross validation
    
    INPUTS
    - Nsim: number of arrays of training set indices
    - Training_prop: proportion of the dataset that will be included 
      in the training set
    - method: 'random', the indices are randomly selected without replacement,
              'block', the training indices are the int(Training_prop*N)
                      first indices
                      
    OUTPUT
    List of Nsim arrays of training indices
    '''
    
    indref = np.arange(N)
    Ntrain = int(N*Training_prop)
    training_indices = []
    
    if method == 'random':
        
        for i in range(Nsim): 
        
            indpermut = np.random.permutation(indref)
            indtrain = indpermut[0:Ntrain]
            indtest = indpermut[Ntrain:]
            indtrain = np.sort(indtrain)
            indtest = np.sort(indtest)
            
            training_indices.append([indtrain, indtest])
            
    if method == 'block':
        
        Ntry = min(Nsim, int((1-Training_prop)*N))
        for i in range(Ntry): 
        
            indtrain = indref[i:i+Ntrain]
            indtest = np.concatenate((indref[:i],indref[i+Ntrain:]))
            
            training_indices.append([indtrain, indtest])
            
        
    return(training_indices)



      
    
def crossval(T,y0,W2, omega_peaks, Mat, Nsim = 200,  
                                     method='random',
                                     Training_prop=0.6):
    
    '''
    Computes the cross validation score of the couple (W2, omega_peaks, Mat), 
    that is the noise model, frequencies and null hypothesis model. A linear model 
    [Mat, cos(omega_peaks[0] * T), sin(omega_peaks[0] * T), ..., cos(omega_peaks[-1] * T), sin(omega_peaks[-1] * T)]
    where cos(omega_peaks[0] * T) and sin(omega_peaks[0] * T) are column cectors
    is fitted onto training sets defined by training_indices and the likelihood
    of the predicted model is computed on the test sets. 
    
    INPUTS:
    - T: time stamps (np array)
    - y0: data (np array of same size as T)
    - W2: inverse of the covariance matrix
    - omega_peaks: np array of frequencies of signals assumed to be in the data
    - Mat: linear model of the null hypothesis len(T) x p, p>0
    - training_indices: List of Nsim arrays of training indices
    OUTPUT
    - meanLikelihood: mean value of the likelihood evaluated on the Nsim test sets
    - medianLikelihood: median value of the likelihood evaluated on the Nsim test sets
    - loglikelihood_all_data: likelihood of the model on the whole timeseries
    '''
    N = len(T)
    training_indices = training_sets(N, Nsim=Nsim, 
                                     Training_prop = Training_prop, 
                                     method = method)
   # Nsim = len(training_indices)    
    loglikelihoods = np.zeros(Nsim)
  
    L = np.linalg.cholesky(W2)
    chi2_all = residuals(T, y0, W2, omega_peaks, Mat, return_only_chi2 = True)
   
    halflogdet_all_data = np.sum(np.log(np.diag(L)))
    C_all = 0.5*len(T)*np.log(2*np.pi)

    loglikelihood_all_data = - 0.5*chi2_all  \
                     - C_all \
                     + halflogdet_all_data
                     
    #print('training_indices  ',len(training_indices))
    
    for i in range(Nsim): 
        

        indtrain = training_indices[i][0]
        indtest = training_indices[i][1]
        Ntest = len(indtest)
        C = 0.5*Ntest*np.log(2*np.pi)
        
        Ttrain = T[indtrain]
        y0train = y0[indtrain]
        W2train = W2[:,indtrain]
        W2train = W2train[indtrain,:]
       
        Ttest = T[indtest]
        y0test = y0[indtest]
        W2test = W2[:,indtest]
        W2test = W2test[indtest,:]    
        
        Mattrain = Mat[indtrain,:]
        Mattest= Mat[indtest,:]
        
        restrain, xtrain = residuals( Ttrain, y0train, 
                                 W2train, omega_peaks, 
                                 Mattrain, return_matrix = False)
        
        
        restest,xtest, Atest = residuals( Ttest, y0test, 
                                 W2test, omega_peaks, 
                                 Mattest, return_matrix = True)    
        
        model_prediction = Atest.dot(xtrain)
        respred = y0test - model_prediction 
        Ltest = np.linalg.cholesky(W2test)
        aux = Ltest.dot(respred)
        chi2_prediction = aux.dot(aux)
        halflogdet = np.sum(np.log(np.diag(Ltest)))
        
        loglikelihood_prediction = - 0.5*chi2_prediction  \
                                -  C \
                                + halflogdet #Det of W2 not covariance mat
             
                                
        loglikelihoods[i] = loglikelihood_prediction
        
    meanlogLikelihood = np.mean(loglikelihoods)
    medianlogLikelihood = np.median(loglikelihoods)
    
    return(meanlogLikelihood,medianlogLikelihood, loglikelihood_all_data)


    
    




    