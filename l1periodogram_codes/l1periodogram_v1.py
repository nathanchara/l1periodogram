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
import matplotlib.pyplot as plt
import time
#import resource

import significance
import covariance_matrices
import lars_l1p
import gglasso_basis_pursuit_l1p
#import gglasso_basis_pursuit_test



# ------------------------------------------------------------------- #
# ------------------------- Initialization ------------------------- #
# ------------------------------------------------------------------- #



class l1p_class():
    def __init__(self, T_init, y):#,*args, **kwargs
        
        self.y_init = y # time series of radial velocities
        self.y = y - np.mean(y) # time series centered for numerical stability
        self.t_init = T_init  # measurement epochs of radial velocities
        self.t = T_init - np.mean(T_init) # measurement epochs centered, for numerical stability
        self.Tobs = T_init[-1] - T_init[0] # observation timespan
        self.Nt = len(T_init) # number of observations

        self.starname = '' # name of the star studied
        self.dataset_names = ['data'] # list of the different data sets (for instance if there are several instruments)
        
        self.omegas = None # grid of frequencies on which the dictionary is computed (in rad/day)
        self.oversampling = None# deltaOmega is 2*np.pi/self.Tobs/self.oversampling omegas is np.arange(deltaOmega, self.omegamax, deltaOmega)
        self.omegamax = None #Maximum of omegas
        self.weights = None # dictionary weights
        self.periods = None # = 2*np.pi/self.omegas
        
        self.V = np.eye(self.Nt) #covariance matrix
        self.cov_matrix_bound = 1e-8 # elements of the covatiance matrix such that 
        self.W2 = None # inverse covariance matrix
        self.W = None # square root of the inverse covariance matrix
        self.projmat = None #projection matrix
        
        self.offsets = None # matrix of size self.Nt x number of different offsets
        self.MH0 = None #projection matrix
        
        self.dict = None #dictionary used in the minimization
        self.A = None #dictionary whose columns are only sines and cosines
        self.Nphi = None #number of phases corresponding to each frequency in the dictionary 
        
        #minimization_input and minimization_output are dictionaries,
        #each key is a numerical method and
        #each value is a dictionary containting the parameters
        #of the method        
        self.minimization_output = {} 
        self.minimization_input= {} 
        self.bp_tolerance = 0.0
        self.numerical_method = 'lars'

        #indices of the dictionary columns whose coefficients are non zero
        self.active_set = []
        self.coefficients = [] # result of the basis pursuit minimizations 

        self.normalization_factor = 0 # factor intervening in thesmoothing
        self.smoothed_solution = None        
        
        #Peaks of the smoothed lasso solution
        self.omega_peaks = None  # frequency of the peaks
        self.peakvalues = None # height of the peaks
        
        #Dictionary containing the significance evaluation results
        self.significance = {}



# ------------------------------------------------------------------- #
# -------------------------- Main function -------------------------- #
# ------------------------------------------------------------------- #



    def l1_perio(self,
                 tolfact = 1.0, 
                 smoothwidth_factor = 0.5, 
                 verbose = 1, 
                 numerical_method = 'lars',
                 Maxiter = 2000,
                 significance_evaluation_methods = ['fap','evidence_laplace'],
                 max_n_significance_tests = 10,
                 plot_convergence_tests = False,
                 plot_output = True,
                 reweight_args = {'Nit':0, 'q':0.5, 'eps':1e-6},              
                 **kwargs):
        
        '''
        Computes the l1 periodogram with dictionary self.dict
        as described in Hara, Boué, Laskar, Correia (2017) MNRAS
        
        
        Denoting by A := self.A, y:= self.y and W = self.W
        (i)   The code solves the Basis pursuit problem (Chen & Donoho 1998)
       
          (1) {x^*, u^*} = arg min_ {x,u} ||x||_l1 
                    subject to 
          ||W.dot(A.dot(x) + MH0.dot(u) - y)||_l2 <= epsilon_0 * tolfact
        
        where epsilon_0 = np.sqrt(len(y) - MH0.shape[1] - 2)
        with the numerical method self.numerical_method 
        
        (ii)  x^* is smoothed with the self.smooth_solution method
        
        (iii) The significance of l1 periodogram peaks is assessed
        
        Inputs:
            - tolfact (float): see eq. 1 
            - smoothwidth_factor (float): for each frequency omega in the grid 
            self.omegas, the coefficients of x^* corresponding to 
            omega +/- smoothwidth_factor*Delta omega are averaged
            where Delta omega = 2*np.pi / self.Tobs
            - verbose: 0: no printed output
                       1: the main steps are printed
                       2: as 1 plus prints also intermediate outputs of the minimization methods
            - numerical_method (string): method used to solve problem (1) ('lars' or 'gglasso')
            - significance_evaluation_methods (list): method to evaluate the significance of the l1 periodogram peaks
            - max_n_significance_tests (integer): max number of peaks for which the significance is evaluated
            - plot_convergence_tests (bool): plot figures relative to the convergence of the numerical method to solve (1)
            - plot_output (bool): plot the l1 periodogram computed (methods plot_clean and plot_with_list)
            - reweight_args (Python dictionary): reweighting iterations to solve (1) but with lq norm in place of l1, 0<q<1
              if q is set to 0, the l1 norm is replaced by the sum of logarithm of absolute values
              available only when setting numerical_method to gglasso.
        
        Outputs:
            - self.minimization_input (Python dictionary): inputs of the numerical methods 
            - self.minimization_output (Python dictionary): outputs of the numerical methods 
            - self.active_set: components of x^* that are non zero
            - self.coefficients: value of x^* on the active_set indices
            - self.smoothed_solution (np array): value of the l1 periodogram
            - self.significance (Python dictionary): information relative to the significance 
                                                    of the l1 periodogram peaks
        '''
        
        if verbose >=1:
            print('--------- l1 periodogram: start ---------')         
        #if verbose >=1:
            #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        start_time_l1p = time.time()

        # ------- Some initializations ------- #
        Nomega = len(self.omegas)                    
        dim_H0 = np.shape(self.MH0)[1]            
        self.numerical_method = numerical_method
        
        # ------- Algorithm parameters ------- #
        bp_tolerance = np.sqrt(self.Nt-dim_H0-2)*tolfact
        self.bp_tolerance = bp_tolerance
   
        # ------- Algorithm input ------- #   
        if verbose >=1: 
            print('Setting minimization input')
        y_res,x = significance.residuals(self.t, self.y, self.W2, [], self.MH0)
        Wy_res = self.W.dot(y_res)
        Wy_res_c = Wy_res - np.mean(Wy_res)
        normy = np.linalg.norm(Wy_res_c)
        y_input = Wy_res_c/normy   
        #normalization of the input data (useful for the smoothing)
        self.normalization_factor = normy
        
        # ------ Define the dictionary ------ #
        if verbose >=1: 
            print('Weighting the dictionary')
            start_time = time.time()
        self.weight_dict_sines()   
        if verbose >=1: 
            print('Weighting dictionary duration', time.time() - start_time, "seconds")
     
        # ------- Compute the Lasso solution ------- #  
        if self.numerical_method =='lars':
            #LARS requires centered and normalized columns
            self.center_dict()
            if verbose >=1:
                print('Dictionary columns centred')            
            self.normalize_dict()
            if verbose >=1:
                print('Dictionary columns normalized')             
        
            # ------- LARS input ------- #            
            tol_lars_input = bp_tolerance/normy
            
            # ------- Store input ------- #
            self.minimization_input['lars'] = \
            {'y_lars_input': y_input, 
             'tol_lars_input':tol_lars_input}

            # ------- Compute the LARS ------- #
            if verbose >=1:
                print('Computing LARS path')
                start_time = time.time()
            dict_out =\
            lars_l1p.lar(y_input,self.dict,lasso=1,
                           tol=tol_lars_input,
                     verbose=verbose,maxiter=Maxiter)          
            if verbose >=1:
                print('LARS duration', time.time() - start_time, "seconds")
            
             # ------- Store output ------- #
            self.minimization_output['lars'] = dict_out
            self.active_set = dict_out['active_set']
            self.coefficients = dict_out['betas']
            
            # -------  Convergence tests ------- #
            if plot_convergence_tests:
                self.plot_convergence_tests('lars')
                
        elif self.numerical_method == 'gglasso': 
            
            if self.Nphi != 2:
                print('For gglasso Nphi must be set to 2, recomputing the dictionary')
                self.update_model(Nphi=2)
                self.weight_dict_sines()
            
            # Our implementation of gglasso  is more stable 
            # with centered columns and requires
            # the vectors corresponding to groups 
            # to be orthonormalized
            self.center_dict()
            if verbose >=1:
                print('Dictionary columns centred')   
            self.orthonormalize_2d_dict(verbose=verbose)
            if verbose >=1: 
                print('Dictionary columns orthonormalized')
            
            # ------- Store input ------- #
            self.minimization_input['gglasso'] = \
                {'y_gglasso_input': y_input}
             
            # ------- Solve the group LASSO ------- #
            if verbose >=1:
                print('Computing the group lasso solution (gglasso)')
                start_time = time.time()
            
            bp_tol_in = bp_tolerance/normy
            dict_out = gglasso_basis_pursuit_l1p.gglasso(y_input, 
                                  self.dict,
                                  bp_tol_in, 
                                  bp_bound_numerical_tol = bp_tol_in/20,
                                  beta_init=None, 
                                  weights=None, 
                                  maxit='default', 
                                  tol=1e-5, 
                                  intercept=True,
                                  verbose=verbose,
                                  **kwargs)
            if verbose >=1:
                print('gglasso duration', time.time() - start_time, "seconds")
           
            # Reweighting (lq minimization)
            for i in range(reweight_args['Nit']):
                if verbose >=1:
                    print(' --- reweighting iteration {}/{} --- '.format(i+1,reweight_args['Nit']))
                eps = reweight_args['eps']/(i+1)
                bp_tolerance_reweight = np.sqrt(bp_tolerance**2 - 0.5*bp_tolerance)
                bp_tolerance_in = bp_tolerance_reweight /normy #np.sqrt(reweight_args['Nit']*1.8/(i+reweight_args['Nit']))
                abscoeff = np.sqrt(np.abs(dict_out['beta'][0::2,-1])**2 + np.abs(dict_out['beta'][1::2,-1])**2)
                self.weights = 1/(abscoeff + eps)**(1-reweight_args['q'])
                dict_out = gglasso_basis_pursuit_l1p.gglasso(y_input, 
                                                 self.dict,
                                      bp_tolerance_in, 
                                      beta_init=dict_out['beta'][:,-1], 
                                      weights=self.weights, 
                                      maxit='default', 
                                      tol=1e-5, 
                                      intercept=True,
                                      verbose=verbose)                
            #gc.enable()
            
            # ------- Store output ------- #
            self.minimization_output['gglasso'] = dict_out            
            self.coefficients = dict_out['beta'][:,-1]
            index = np.arange(Nomega*2)
            trutharr = self.coefficients != 0
            self.active_set = index[trutharr]
            self.coefficients = self.coefficients[trutharr]
        
        # -------  Smooth solutions ------- #
        if verbose >=1:
            print('Smoothing lasso solution')
            start_time = time.time()
        self.smooth_solution(smoothwidth_factor=smoothwidth_factor, **kwargs)
        if verbose >=1:
            print('Smoothing duration ', time.time() - start_time, "seconds")
                                      
        # Compute at which frequencies the peaks are attained    
        start_time = time.time()
        self.find_peaks()   
        if verbose >=1:
            print('Find peaks duration', time.time() - start_time, "seconds")
        
        # ------------- Plot smoothed solution ------------- #
        if plot_output:
            start_time = time.time()
            N_markers = min(len(self.omega_peaks),
                            max_n_significance_tests)  
            self.plot_clean(N_markers,
                            annotations = 'periods',
                            save = True,
                            **kwargs)            
        if verbose >=1:
            print('Plot (clean) duration ', time.time() - start_time, "seconds")
            
        # ------------- Compute peaks significance ------------- #
        if verbose >=1:
            print('Evaluating peaks significance')
        start_time1 = time.time()                
        self.evaluate_significance(max_n_significance_tests,
        significance_evaluation_methods = significance_evaluation_methods,
                                   verbose=verbose,
                                  **kwargs)
        
        if verbose >=1:
            print('Significance evaluation duration', time.time() - start_time1, "seconds")
   
        # ------------- Plot l1 periodogram + significance ------------- #
        if plot_output:
            start_time = time.time()            
            self.plot_with_list(N_markers,
                               significance_values = 'log10faps',
                               save = True,
                               **kwargs) 
            if verbose >=1:           
                print('Plot (with list) duration ', time.time() - start_time, "seconds")
                
        if verbose>=1:
            print('--------- l1 periodogram: end ----------')
            print('Total time: ', time.time() - start_time_l1p, ' seconds')
            print('----------------------------------------')



# ------------------------------------------------------------------- #
# ------------------ Dictionary related functions ------------------- #
# ------------------------------------------------------------------- #



    def set_model(self,omegamax = 1.9*np.pi,
                       oversampling = 10, Nphi = 8,
                       V=None, 
                       MH0 = None,
                       weights=None,
#                       numerical_method = 'lars', 
                       verbose=1,**kwargs):

        self.update_model(omegamax = omegamax,
                       oversampling = oversampling, Nphi = Nphi,
                       V=V, 
                       MH0 = MH0,
                       weights=weights,
#                       numerical_method = numerical_method, 
                       verbose=verbose)
        

    def update_model(self,omegamax = None,
                       oversampling = None, Nphi = None,
                       V=None, 
                       MH0 = None,
                       weights=None,
                       #numerical_method = None, 
                       verbose=1,**kwargs):
        
        recompute_dict_sine = False
        #recompute_dict = False
        
        if verbose >=1:
            start_time = time.time()
            print('Creating dictionary')
    
        self.Tobs = self.t[-1] - self.t[0]
        

        if omegamax is not None:
            #If only the max frequency changes, no need to recompute the whole dictionary
            if self.omegamax is not None:
                allconst = oversampling is None and V is None \
                    and Nphi is None and V is None and MH0 is None \
                    and weights is None #and numerical_method is None
                if omegamax <= self.omegamax and allconst:
                    
                    i = np.argmin(np.abs(self.omegas - omegamax))
                    self.omegas = self.omegas[:i]
                    self.periods = 2*np.pi/self.omegas
                    self.dict = self.dict[:,:i*self.Nphi]
                    self.A = self.A[:,:i*self.Nphi]
                    if verbose >=1:
                        print('Max frequency updated')
                else:
                    recompute_dict_sine = True
                    #recompute_dict = True
            else:
                recompute_dict_sine = True
                #recompute_dict = True                
            self.omegamax = omegamax
        
        if oversampling is not None:
            self.oversampling = oversampling
            recompute_dict_sine = True
            #recompute_dict = True

        if Nphi is not None:
            self.Nphi = Nphi
            recompute_dict_sine = True
            #recompute_dict = True
        
            
        if V is not None: 
            self.V = V
            self.create_precision_matrix()
            #recompute_dict = True
            
        if MH0 is not None:
            self.MH0 = MH0
            self.create_projmat()
            #recompute_dict = True
            if verbose >=1:
                print('Projection matrix set, dimension of the unconstrained space (number of MH0 columns): {}'.format(self.MH0.shape[1]))

        if weights is not None:
            self.weights = weights
            #recompute_dict = True

        #Recompute the dictionary as needed
        if recompute_dict_sine:  
            self.create_freqgrid(self.omegamax, self.oversampling)
            self.create_dict_sine(self.Nphi)    
            self.periods = 2*np.pi/self.omegas
            
        if verbose >=1:
            print('Model creation duration', time.time() - start_time, "seconds") 
        
       
            
    def create_freqgrid(self, omegamax, oversampling):
        '''
        Create a grid of equispaced frequencies, 
        frequency spacing is equal to deltaOmega = 2*np.pi/self.Tobs/oversampling
        and the grid spans from deltaOmega to omegamax
        '''
        Nomega = int(omegamax/(2*np.pi)*self.Tobs*oversampling)
        self.omegas = np.arange(1,Nomega+1) / \
                      Nomega*omegamax
        
    
    
    def create_precision_matrix(self, **kwargs):
    #Invert covariance matrix 
        absV = np.abs(self.V)
        sumdiag = np.sum(np.diag(absV))
        sumV = np.sum(absV)   
        if (sumV - sumdiag)/sumV > self.cov_matrix_bound:
            W2,W = covariance_matrices.invert_covmatrix(self.V,
                                              self.cov_matrix_bound, 
                                              **kwargs)
        else:
            W2 = np.diag(1/np.diag(self.V))
            W = np.diag(1/np.sqrt(np.diag(self.V)))
            
        self.W = W
        self.W2 = W2
        
        
    #def initialize_offsets(self, **kwargs): #If not specified data is assumed 
                                      #to come from a single instrument


    def create_dict_sine(self,Nphi):
        ''' 
        create a matrix A whose columns are 
        cos(om * self.t + phi) with all possible combinations of 
        om and phi where om is an element of the 
        frequency grid (self.omegas) and 
        phi is in (k*np.pi/Nphi)_k=0..Nphi-1
        '''
        
        Nomega = len(self.omegas)
        phis = np.arange(Nphi)*np.pi/Nphi 
        cosphi = np.zeros(Nphi)
        sinphi = np.zeros(Nphi)
        for jj in range(Nphi):
            cosphi[jj] = np.cos(phis[jj]) 
            sinphi[jj] = np.sin(phis[jj])
            
#        omT = np.einsum('i,j->ij', self.t,self.omegas)
#        cosomegas = np.cos(omT)
#        sinomegas = np.sin(omT)     
        self.A = np.zeros((self.Nt,Nomega*Nphi))
        for ii in range(Nomega) :
#                cosom =cosomegas[:,ii]
#                sinom =sinomegas[:,ii]
            omega = self.omegas[ii]
            cosom = np.cos(omega*self.t)
            sinom = np.sin(omega*self.t) 
            for jj in range(Nphi):
                v =  cosom*cosphi[jj] -  sinom*sinphi[jj]              
                self.A[:,jj + ii * Nphi] = v  
                                
    
    def create_projmat(self):
        ''' 
        create a projection matrix on the 
        space orthogonal to self.MH0
        ''' 
        
        if self.MH0 is not None:
            WM = np.dot(self.W,self.MH0)
            WMt = WM.T
            WMtWM = np.dot(WMt,WM)
            L = np.linalg.cholesky(WMtWM)
            target = WMt.dot(self.W)
            aux = np.linalg.solve(L,target)
            LS = np.linalg.solve(L.T, aux)
            P = self.MH0.dot(LS)
            ImP = np.eye(self.Nt) - P
            self.projmat = ImP            
        else:
            self.projmat = None
            


    def weight_dict_sines(self):
        # computes W.dot(self.dict_sines), 
        if self.projmat is not None:
            self.dict = self.W.dot(self.projmat.dot(self.A))
        else:
            self.dict = self.W.dot(self.A)
            
            
    def center_dict(self):
        '''
        subtracts the mean of each column of self.dict  and normalizes them
        Center and normalize the columns
        '''
        self.dict = self.dict -  self.dict.mean(axis=0).reshape(1, -1)
        
    def normalize_dict(self):
        '''
        Normalize the dictionary columns indexed by 'index_list'
        Normalizes all columns by default
        if subset is None:
        '''
        Ndict = self.dict.shape[1]
        if self.weights is None or len(self.weights) != Ndict:
            self.weights = np.ones(Ndict)
            
        factors = np.zeros(Ndict)
        A_norms = np.sum(self.dict**2,axis=0)**(1./2)
        index = A_norms>1e-15
        #print(len(index), len(self.weights), len(A_norms))
        factors[index] = self.weights[index]/A_norms[index] 
        self.dict[:,index] = np.einsum('ij,j->ij',  self.dict[:,index],
              factors[index])

    def orthonormalize_2d_dict(self, verbose=1):
        '''
        Orthonormalize the columns of the dictionary two by two 
        (each two consecutive columns are orthonormalized)
        '''
        
        if self.Nphi != 2:
            raise Exception('Nphi must be set to two to orthonormalize the dictionary, use c.update_model')
        
        #objgraph.show_backrefs(random.choice(objgraph.by_type('spell')),filename="spell_refs.png")
        #objgraph.show_refs(self.dict, filename='sample-graph.png')   
        
        #print('in spell: normalize cosines')
        #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)  
        #objgraph.show_most_common_types()
        #Normalize cosines
        cos_norms = np.sqrt(np.sum(self.dict[:,::2]**2,axis=0))
        cos_norms[cos_norms==0] = 1
        self.dict[:,::2] = np.einsum('ij,j->ij',  self.dict[:,::2], 1/cos_norms)     
        
        #print('in spell: orthonormalize')
        #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        #objgraph.show_most_common_types()
        #orthogonalize sines
        dotprods = np.einsum('ij,ij->j',  self.dict[:,::2], self.dict[:,1::2])    
        correction_sines = np.einsum('ij,j->ij',  self.dict[:,::2], dotprods)     
        self.dict[:,1::2] = self.dict[:,1::2] - correction_sines

        #print('in spell: id(gglasso_final_test)', id(gglasso_basis_pursuit_l1p))
        #print('in spell: id(self.dict)', id(self.dict))
        #print('in spell: normalize sines')  
        #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        #objgraph.show_most_common_types()
        #normalize sines
        
        
        sin_norms = np.sqrt(np.sum(self.dict[:,1::2]**2,axis=0))
        #print('in spell: sine normalization step 1')
        sin_norms[sin_norms==0] = 1 
        #print('in spell: sine normalization step 2')
        self.dict[:,1::2] = np.einsum('ij,j->ij',  self.dict[:,1::2], 1/sin_norms)           
        if verbose >=1:
            print('in spell: orthonormalization done')        
            
            
            
    def unpenalize_periods(self,periods,MH0_core,**kwargs):
        '''
        Add cosine at periods 'periods' 
        to the matrix of unpenalized vectors MHO_core
        Inputs:
            - periods: list of periods whose coefficients will be unpenalized
              in the l1 minimization
            - MH0_core: other vectors that are unpenalized
        Output:
            - updates self.dict with the new list of unpenalized vectors
        '''
        
        n0 = MH0_core.shape[1]
        nper = len(periods)
        M1 = np.zeros((self.Nt, n0+nper*2))

        
        M1[:,:n0] = MH0_core
        for i in range(nper):
            om = 2*np.pi/periods[i]
            cosv = np.cos(om*self.t)
            sinv = np.sin(om*self.t)
            M1[:,n0+i*2]   = cosv/np.linalg.norm(cosv)
            M1[:,n0+i*2+1] = sinv/np.linalg.norm(sinv)
        
        self.update_model(MH0=M1,**kwargs)
        
    def show_dict_parameters(self):
        '''
        Show the parameters of the current dictionary
        '''
            
        print('Dictionary parameters')
        print('Max frequency in the frequency grid, omegamax:', self.omegamax, 'rad/day, corresponding to a period ', 2*np.pi/self.omegamax, ' d')
        print('Oversampling of the frequency grid, oversampling:', self.oversampling)
        print('Number of phases per frequency, Nphi:', self.Nphi)
        print('Dictionary set for the numerical_method matrix:', self.numerical_method)
        print('Unpenalized vectors, MH0: ')
        print(self.MH0)
        
        plt.figure()
        absV = np.abs(self.V)
        index = absV <1e-16
        absV[index] = np.nan
        plt.imshow(np.log10(absV))
        plt.suptitle('Noise covariance matrix (attribute V)', fontsize = 20)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('log10 value',fontsize = 18)



# ------------------------------------------------------------------- #
# ----------------------- Smoothing functions ----------------------- # 
# ------------------------------------------------------------------- #


    
    def smooth_solution(self,smoothwidth_factor=0.5,
                        smoothing = 'fit',**kwargs):
        '''
        Smoothes the output of the basis pursuit algorithm
        '''
        #Instantiates the attribute self.smoothed_solution
        Nomega = len(self.omegas)
        omegamax = np.max(self.omegas)
        xx = np.zeros(self.Nphi*Nomega)
        xx[self.active_set] = self.coefficients
        smooth_sol = np.zeros(Nomega)
        
        smoothwidth = max([int(np.floor(smoothwidth_factor*2*np.pi
                            /self.Tobs*Nomega/omegamax)),1])
        #vect0 = self.t*0
        #halfwidth = np.int(width/2)
    
        # Define the grid indices for which the smoothing is computed
        #active_omegas = np.mod(self.active_set, Nomega)
        active_omegas = np.floor(np.array(self.active_set)/self.Nphi).astype(int)
        index_smooth = []
        for i in active_omegas:
            ind1 = max(i - smoothwidth , 0)
            ind2 = min(i + smoothwidth , Nomega-1)
            index_smooth = index_smooth + list(np.arange(ind1,ind2)) 
        
        
        if smoothing == 'fit':
            for it in index_smooth:
                om = self.omegas[it]
                matT = np.array([np.cos(om*self.t), np.sin(om*self.t)]).dot(self.W)
                mat = matT.T
                G = np.dot(mat.T, mat)
#                vect = vect0
                
#                for jj in range(self.Nphi):)
#                    ind1 = np.max([it - width + jj*Nomega], 0)
#                    ind2 = np.min([it + width + jj*Nomega, self.Nphi*Nomega-1])
#                    vect = vect + self.dict[:,ind1:ind2].dot(xx[ind1:ind2])

                ind1 = np.max([(it - smoothwidth)*self.Nphi], 0)
                ind2 = np.min([(it + smoothwidth)*self.Nphi, self.Nphi*Nomega-1])
                #stop
                vect =  self.dict[:,ind1:ind2].dot(xx[ind1:ind2])

                matTvect = np.dot(matT,vect)
                params = np.linalg.solve(G, matTvect)
                smooth_sol[it] = np.sqrt(params[0]**2 + params[1]**2)

        if smoothing == 'sum':
            #Quicker but less precise
            for it in index_smooth:

                ind1 = np.max([(it - smoothwidth)*self.Nphi], 0)
                ind2 = np.min([(it + smoothwidth)*self.Nphi, self.Nphi*Nomega-1])
                smooth_sol[it] = np.abs(np.sum(xx[ind1:ind2])) #np.sqrt(np.sum(xx[ind1:ind2])**2)
                
        self.smoothed_solution = smooth_sol*self.normalization_factor
            
            
    def find_peaks(self):
        '''
        define the peaks appearing in the smoothed solution
        solution: their frequencies self.omega_peaks
        their values self.peakvalues, and decreasing order
        self.indexsort
        '''
        i = 0

        peakpos = []
        Nomegas = len(self.omegas)
        
        while i<Nomegas-1:
            if self.smoothed_solution[i] !=0:
                j = i
                while self.smoothed_solution[i] !=0 and i<Nomegas-1:
                    i = i+1
                array = self.smoothed_solution[j:i]
                peakpos1 = j+np.argmax(array)
                peakpos.append(peakpos1)
            i = i+1
    
        peakvals = self.smoothed_solution[peakpos]
        indexsort = np.argsort(-peakvals)
        peakpos = np.array(peakpos)
        peakpos_sort = peakpos[indexsort]
        
        if len(peakpos_sort)>0:
            self.omega_peaks = self.omegas[peakpos_sort]
            self.peakvalues = self.smoothed_solution[peakpos_sort]
        else:
            self.omega_peaks = []
            self.peakvalues = []

    def peak_values_at(self,list_periods):
        '''
        finds the value of the smoothed solution
        whose corresponding periods are closest to 
        list periods
        '''
        nper = len(list_periods)
        peakvalues = np.zeros(nper)
        for i in range(nper):
            om = 2*np.pi/list_periods[i]
            ind = np.argmin(np.abs(self.omegas - om))
            peakvalues[i] = self.smoothed_solution[ind]
        return(peakvalues)



# ------------------------------------------------------------------------ #
# ----------------- Evaluation of the peaks significance ----------------- #
# ------------------------------------------------------------------------ #

    
    
    def evaluate_significance(self, max_n_significance_tests,
            significance_evaluation_methods = ['fap','evidence_laplace'],
                               fap_computation='baluev2008',
                               verbose=1,
                               **kwargs):
        '''
        Evaluate the significance of peaks found by the l1 periodogram
        '''
        
        #Check that the number of statistical tests does not
        #exceed the number of frequencies with nonzero peaks
        number_nonzero_peaks = len(self.omega_peaks)
        Ntests=np.min([max_n_significance_tests,number_nonzero_peaks])       
        Ntests = int(Ntests)
        if 'fap' in significance_evaluation_methods:
            self.significance['log10faps'] = np.zeros(Ntests)
        if 'evidence_laplace' in significance_evaluation_methods:
            self.significance['log_evidences_laplace']  = np.zeros(Ntests+1)
            self.significance['log_bayesf_laplace']  = np.zeros(Ntests)
            self.significance['log10_bayesf_laplace']  = np.zeros(Ntests)
            self.significance['log_evidences_laplace'][0] = significance.logEv_omegas(
                                     self.t,self.y,self.W2,
                                     [], self.MH0)
        if self.MH0 is not None:    
            nH = self.MH0.shape[1]
        else:
            nH = 0
        omegamax = np.max(self.omegas)
        
        res,x = significance.residuals(self.t, self.y, self.W2, [], self.MH0)
        if number_nonzero_peaks >0:
            chi20 = np.dot(self.W2,res)
            chi20 = np.dot(res.T, chi20)
            
            err1 = np.sqrt(np.diag(self.V))
            #Mstar = 1
            for i in range(Ntests):
                if verbose >=1:
                    print('Significance evaluation', i+1, '/', Ntests)
                #print('Init omega_peaks', self.omega_peaks[0:i+1])
                
                chi2K, omegas_fitted = significance.residuals_nl(self.t,
                                             self.y, self.W2,
                                             self.MH0,
                                        self.omega_peaks[0:i+1],
                                        method= 'L-BFGS-B')
                
                if 'fap' in significance_evaluation_methods:
                    power = (chi20 - chi2K)/chi20
                    if fap_computation =='baluev2008':
                        self.significance['log10faps'][i] =\
                        significance.fap(self.t, err1,
                                   omegamax, power, nH)
                    else:       
                        self.significance['log10faps'][i]  = significance.fapV(self.t, self.W2, 
                                                omegamax, power, 
                                                nH)
                    chi20 = chi2K
                    nH = nH + 2
                
                if 'evidence_laplace' in significance_evaluation_methods:
                
                    self.significance['log_evidences_laplace'][i+1] = significance.logEv_omegas(
                                     self.t,self.y,self.W2,
                                     omegas_fitted, self.MH0)
                    logbf = self.significance['log_evidences_laplace'][i+1] - \
                    self.significance['log_evidences_laplace'][i]
                    self.significance['log_bayesf_laplace'][i] = logbf
                    self.significance['log10_bayesf_laplace'][i] = logbf/np.log(10)
 
                   

# ------------------------------------------------------------------- #
# ------------------------- Plot functions ------------------------- # 
# ------------------------------------------------------------------- #


    
    def plot_input_data(self,save=False, **kwargs):
        '''
        plot the data in the class
        '''
        Nt = len(self.t)
        if self.offsets is None:
            self.offsets = np.ones((Nt,1))
        ninst = self.offsets.shape[1]
        if ninst != len(self.dataset_names):
            self.dataset_names = []
            for i in range(ninst):
                self.dataset_names.append('data_{}'.format(i)) 
        err = None   
        if self.V.shape == (Nt,Nt) and np.linalg.norm(self.V - np.eye(Nt))>0:
            err = np.sqrt(np.diag(self.V))
        
        fig = plt.figure(figsize=(8, 6))
        for i in range(ninst):
            ind = self.offsets[:,i] == 1
            lab = self.dataset_names[i] + ', {} data points'.format(np.sum(ind))
            if err is None:
                plt.plot(self.t[ind], self.y[ind], 'o',
                     label = lab)
            else:
                plt.errorbar(self.t[ind], self.y[ind],
                             yerr=err[ind], marker='o',
                     label = lab,linestyle='')  
            
        plt.xlabel('Time (day)',fontsize=16)
        plt.ylabel('RV (m/s)',fontsize=16)  
        title = self.starname + ', {} data points'.format(np.sum(Nt))
        fig.suptitle(title,fontsize=18)
        plt.legend()
        if save:
            plt.savefig(self.starname +'_timeseries.pdf', format='pdf', dpi=1200)
    
    
    #def center_normalize(self,vect,**kwargs):
        
        #return(vect_cn)
        
         
    def plot_convergence_tests(self,numerical_method):
        '''
        Plots to diagnose convergence of the numerical methods
        '''

        if numerical_method =='lars':
            
            fig = plt.figure()
            #plt.plot(self.t,,'o',label='data')
            plt.plot(self.t,self.minimization_output['lars']['model'],'o', label='lars')       
            plt.xlabel('Time (rad/day)',fontsize=16)
            plt.ylabel('Amplitude (m/s)',fontsize=16)
            fig.suptitle(self.starname + ' Output lars model',fontsize=16)
            plt.legend()
            
            fig = plt.figure()
            plt.plot(self.minimization_output['lars']['normres_arr'], label='lars')
            plt.xlabel('Iteration number',fontsize=16)
            plt.ylabel('Norm of residuals',fontsize=16)
            fig.suptitle(self.starname + ' Evolution of the residuals norm',fontsize=16)    


       

    def plot_with_list(self,number_highlighted_peaks_in,
                      significance_values='log10faps',
                      title = 'default',
                      marker_color = (0.85,0.325,0.098),
                      save = False,
                      **kwargs):

        nmaxpeaks = len(self.omega_peaks)
        number_highlighted_peaks = min(number_highlighted_peaks_in,nmaxpeaks)
        if number_highlighted_peaks_in >  nmaxpeaks:
            print('There are only', nmaxpeaks, 'peaks')

        #significance_values is a key of the dictionary self.significance
        if len(self.omega_peaks[:number_highlighted_peaks])>0:
            periods_plot = 2*np.pi/self.omega_peaks[:number_highlighted_peaks]
        else:
            periods_plot =[]
        peakvalues_plot = self.peakvalues[:number_highlighted_peaks]
        
        fig = plt.figure(figsize=(10, 5.8 + 0.2*number_highlighted_peaks))
        if title =='default':
            l1perio_title = self.starname + ' l1 periodogram'
        else:
            l1perio_title = title
        if self.smoothed_solution is not None:
            plt.plot(self.periods, self.smoothed_solution, linewidth=2, 
                     label=self.numerical_method,color=(0,0.447,0.741))
    
        if number_highlighted_peaks>0:
            plt.plot(periods_plot, 
            peakvalues_plot,'o', color=marker_color,markersize=7
           ,label = 'Highest peaks')
            plt.legend(fontsize=14)
        plt.xlabel('Period (days)',fontsize=16)
        plt.ylabel('Coefficient amplitude',fontsize=16)
        plt.xscale('log')
        yl1,yl2 = plt.ylim()
        plt.ylim((0,yl2))
        
        fig.suptitle(l1perio_title, fontsize=20,y=0.98)    
        
        if significance_values is None:
            string_sig = ''
        elif significance_values in self.significance.keys():
            string_sig = significance_values
        else:
            raise Exception(('The significance_value key word '
            'has to be a key of the self.significance dictionary '
            'or None'))       
        
        #Cleaner outputs for standard significance evaluations
        if significance_values =='log10faps':
            string_sig = r'$log_{10}$ FAPs'
        elif significance_values == 'log_bayesf_laplace':
            string_sig = r'$log$ Bayes factor'  
        elif significance_values == 'log10_bayesf_laplace':
            string_sig = r'$log_{10}$ Bayes factor'         
    
        s1 = '-------------------------------------------------------------------'
        s2 = 'Peaks at (days)    |     ' +  string_sig
        string = s1 + '\n' + s2 + '\n' + s1
        
        if number_highlighted_peaks>0:
            for i in range(number_highlighted_peaks):
                period = periods_plot[i]
                periodstr = str(period) 
                if significance_values is not None:
                    sig_str = str(self.significance[significance_values][i])
                else:
                    sig_str =''
                string1 = periodstr[0:10] + '        |' + sig_str
                string = string + '\n' + string1
                #print(string1)
                
        ax = fig.axes
        #plt.text(0,1.05, string, transform=ax[0].transAxes, fontsize=18)
        plt.text(0.2,1.05, string, transform=ax[0].transAxes, fontsize=14)
        fraction_of_l1perio_plot = 0.75 - 0.02*number_highlighted_peaks
        plt.gcf().subplots_adjust(top=fraction_of_l1perio_plot)
        
        if save:
            string_save = self.starname.replace(' ', '_') + '_l1_periodogram_period.pdf'
            plt.savefig(string_save, format='pdf', rasterized = True)

    
        
    def plot_clean(self,number_highlighted_peaks_in,
                   annotations='periods',
                   marker_color = (0.85,0.325,0.098),
                   title = 'default',
                   save = False,
                      **kwargs): 
        ''' 
        Plot the l1 periodogram with highlighted highest peaks
        INPUTS:
            - number_highlighted_peaks_in: number of peaks to highlight
            - annotations: labels of the peaks 
            annotations is either set to a key of the 
            dictionary self.significance, and then corresponds to the
            significance of the highlighted peaks or to 'periods',
            in which case it shows the peak periods
            - marker_color: color of the marhers that highlight the 
            highest peaks
            - title: plot title
            - save: saves a pdf file if set to True
        '''
        #start_time = time.time()
        
        nmaxpeaks = len(self.omega_peaks)
        number_highlighted_peaks = min(number_highlighted_peaks_in,nmaxpeaks)
        if number_highlighted_peaks_in >  nmaxpeaks:
            print('There are only', nmaxpeaks, 'peaks')
            
        peakvalues_plot = self.peakvalues[:number_highlighted_peaks]
        
        if title=='default':
            l1perio_title = self.starname + ' ' + r'$\ell_1$' +' periodogram'
        else:
            l1perio_title = title
        
        if len(self.omega_peaks[:number_highlighted_peaks])>0:
            periods_plot = 2*np.pi/self.omega_peaks[:number_highlighted_peaks]
        else:
            periods_plot =[]
        

        bluematlab = (0,0.447,0.741)
        
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.periods, self.smoothed_solution, linewidth=1.7,
                 label=r'$\ell_1$' +' periodogram', color=bluematlab
                 , rasterized = True)
        if number_highlighted_peaks>0:
            periods_maxpeaks = periods_plot
        maxperiod = self.periods[0]
                
        axes = plt.gca()               
        minP = min(self.periods)
        maxP = max(self.periods)
        axes.set_xlim([minP,maxP])
            
        ylim = axes.get_ylim()
        deltaY = ylim[1] - ylim[0]
        deltaX = np.log10(maxP) - np.log10(minP)

        if number_highlighted_peaks>0:
            
            if annotations is None:
                point_label=''          
            elif annotations=='periods':
                point_label = 'Peak periods (d)'
            elif annotations in self.significance.keys():
                point_label = annotations
            else:
                raise Exception(('The annotations key word '
            'has to be a key of the self.significance dictionary '
            'or ''periods'' or None'))
                
            #Cleaner outputs for standard significance evaluations
            if annotations=='log10faps':
                point_label = r'$\log_{10}$' + ' FAPs'
            if annotations=='log_bayesf_laplace':
                point_label = r'$\log$' + ' Bayes factor'
            if annotations=='log10_bayesf_laplace':
                point_label = r'$\log_{10}$' + ' Bayes factor'            
            
            plt.plot(periods_maxpeaks, 
                     peakvalues_plot,'o',
                     markersize=5, color=marker_color,
                     label = point_label)
        
        
        y_legend = np.zeros(number_highlighted_peaks)
        x_legend = np.zeros(number_highlighted_peaks)

        for j in range(number_highlighted_peaks):
            
            if annotations=='periods':
                per = periods_maxpeaks[j]   
                p = str(per)
                indexpoint = p.find('.')
                if indexpoint<3:
                    ndigits = 4
                else:
                    ndigits =   indexpoint
                per = round(per, ndigits - indexpoint)
                p = str(per)
                annotation = p[:ndigits]
                
            elif annotations is not None:
                if j < len(self.significance[annotations]):
                    annotation = sci_notation(self.significance[annotations][j], decimal_digits=1, 
                                     precision=None, exponent=None)
                else:
                    annotation=''
            
            x1 = periods_maxpeaks[j]*1.17
            y1 = peakvalues_plot[j] #- deltaY/10
            if np.log10(np.abs(x1/maxperiod))>0.85:
                x1 = periods_maxpeaks[j]*0.8
            #if np.log10((x1 - minperiod)/minperiod)<0.1
            
            #deltalogp = np.lod10(max(periods) - min(periods))
            diff_y = np.abs(y1 - y_legend[:j])
            diff_x = np.abs((np.log10(x1/x_legend[:j])))
            
            condition = (diff_y <deltaY/9) *  (diff_x<deltaX/9)
            index_cond = [i for i,v in enumerate(condition) if v]
            
    
            indices = np.arange(j)
            if j>0 and np.sum(condition)>0:
                distx = diff_x[index_cond]
                ii = indices[condition][np.argmin(distx)]
                if y1 <y_legend[ii]:
                    if y1>deltaY/10:
                        y1 = y_legend[ii] - deltaY/10
                    else:
                        x1 = x1* 1.1
                else:
                    y1 = y_legend[ii] + deltaY/10
            y_legend[j] = y1 
            x_legend[j] = x1                 
            
            if annotation is not None:
                plt.annotate(annotation, (x1,y1),
                             ha='left', fontsize=16, 
                             color=(0.85,0.325,0.098),
                             bbox=dict(facecolor='white', 
                                       edgecolor = 'white',
                                       alpha=0.8))         
        
        plt.xlabel('Period (days)',fontsize=16)
        plt.ylabel('coefficient amplitude',fontsize=16)
        plt.xscale('log')
        yl1,yl2 = plt.ylim()
        plt.ylim((0,yl2))
        plt.legend(fontsize = 16, loc='best')
        fig.suptitle(l1perio_title ,fontsize=20,y=0.97)    
        
        if save:
            string_save = self.starname.replace(' ', '_') + '_l1_periodogram_notext.pdf'
            plt.savefig(string_save , rasterized = True,
                        format='pdf')




    



def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    written by sodd (stackoverflow user)
    
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if num!=0 and ~np.isnan(num):
        if exponent is None:
            exponent = int(np.floor(np.log10(abs(num))))
        coeff = round(num / float(10**exponent), decimal_digits)
        if precision is None:
            precision = decimal_digits
        if exponent !=0:
            return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)
        else:
            return r"${0:.{2}f}$".format(coeff, exponent, precision)
    else:
        return('0')                
