# -*- coding: utf-8 -*-

# Copyright 2020 Alessandro Mari
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
import gglasso_wrapper as glasso_func
import matplotlib.pyplot as plt


def gglasso(y, X, bp_tol_in, beta_init=None, weights=None,
            maxit="default", tol=1e-4, 
            intercept=True, verbose=0, 
            bp_bound_numerical_tol = 0.01,basis_pursuit_maxiter=50):
    """
    Compute the solution of the group lasso for groups of size 2 at a grid of Lambdas by coordinate majorization descent.
    Solves 1/(2*n)|y-Xbeta|+lambda*sum(|beta_i|) where n is the length of y.
    Based on the article
    AUTHORS:  *Yi Yang(yi.yang6 @ mcgill.ca) and + Hui Zou(hzou @ stat.umn.edu),
              *Department of Mathematics and Statistics, McGill University
              + School of Statistics, University of Minnesota.
    REFERENCES: Yang, Y. and Zou, H.(2015). A Fast Unified Algorithm for Computing Group -
                Lasso Penalized Learning Problems Statistics and Computing. 25(6), 1129 - 1141.

    Parameters
    ----------
    :param y: observations y. Must be a one-dimensional array
    :param X : matrix of predictor. The number of rows and the length of y must be equal
    :param Lambdas : Grid of evaluation (can be a float).  Must be a one-dimensional array.
    :param beta_init : Starting position for the largest Lambdas
    :param weights : Array of weights.  Must be a one-dimensional array. The length of the array should be equal to the number of groups
    :param maxit : Maximum number of iteration allowed
    :param tol : Convergence tolerance for coordinate majorization descent.
    :param epsilon : Controls the majorisation of the curve. Should be small.
    :param intercept : Boolean that include an intercept in the model.
    :param verbose : Boolean, 2 print the results of the fortran iteration, 1 prints checks in this file

    :returns:- active_set: set of indices corresponding to non-zero coefficients for the smallest lambda (last computed),
             - beta0: array corresponding to the intercept at each lambda,
             - beta: matrix whose columns correspond to the solution at each lambda,
             - model: X*beta for the smallest lambda (last computed),
             - Xact: Columns corresponding to the active set,
             - norm_beta: Norm of each group for the smallest lambda  (last computed) ,
             - res: residuals for each lambda (sequence of deacreasing lambda),
             - norm_res: norm of the residuals for each lambda,
             - computed_lam: Array of lambda for which the computation succeed.
    -------

    """

    # Do some check and initialisation -----------------------------------
#    ok, message = check_argument(y, X, Lambdas, beta_init, weights, maxit, tol, epsilon, intercept)
#    if not ok:
#        print(message)
#        return 0, 0, 0, 0, 0, 0, 0, 0, 0
    
#    print('gglasso: input checked')
    #y = np.copy(yin)
    #X = np.copy(Xin)
    yin = np.copy(y)
    nrow = X.shape[0]
    ncol = X.shape[1]
    nbr_group = int(ncol / 2)

    if maxit == "default":
        maxit = int(1000000)

    if weights is None:
        weights = np.ones(nbr_group)
    if beta_init is None:
        beta_init = np.zeros(ncol)
    
    Nt = len(y)
    #Lambdas = np.array([bp_tolerance**2/Nt/normy**2])
    #print(bp_tolerance, Lambdas, Nt)
    correlations = y.T.dot(X)
    correlations_cos = correlations[0::2]
    correlations_sin = correlations[1::2]
    correlations_groups = np.sqrt(correlations_cos**2 + correlations_sin**2)
    lambdamax = np.max(correlations_groups)/Nt
    Lambdas = np.array([lambdamax])
    
    verbose_fortran = np.max(verbose - 1, 0)
    
    
    nvars = ncol
    # End.---------------------------------------------------------------
    if verbose>=1:
        print('gglasso: initilisations done')
    
    # scale y-----------------------------------------------------------

    meany = 0.0
    scaley = 1.0
    if intercept:
        meany = np.mean(y)
        scaley = np.sqrt(nrow) / np.linalg.norm(y - meany)
        y = scaley * (y - meany)
        Lambdas = Lambdas * scaley
        tol *= scaley
        #weights *= scaley 
        
    #print(scaley)
    if verbose>=1:
        print('gglasso: y scaled')

    # Compute eigendecomp for each group-------------------------------
#    stop
#    H = X.T.dot(X)  # the same for the 3 losses
#    gamma = np.zeros(int(ncol / 2))
#    # Compute gamma, i.e. upper bound on the curvature of the Loss
#    for i in range(int(ncol / 2)):  # group of size two
#        print('gglasso decomp',i)
#        H_k = H[2 * i:2 * (i + 1), 2 * i:2 * (i + 1)]
#        D = np.linalg.eigh(H_k)[0]  # eigenvalues
#        eta = D[-1]  # largest eigenvalues
#        gamma[i] = (1 + epsilon) * eta  # for stability
#    gamma /= n
#    print('gglasso: eigen decomposition done')
    
    #each group is orthonormalized
    gamma = np.ones(nbr_group)
        
    # End--------------------------------------------------------------

    # Initialisation param --------------------------------------------
    # Size of each group (here 2,2,...)
    gp_size = np.repeat(2, nbr_group)
    # Starting position of each group (1,3,5,7,...)
    # !!! FORTRAN INDEXES ARRAYS FROM 1 NOT 0 !!! 
    start_val_gp = np.array([2 * i + 1 for i in range(nbr_group)])
    # Ending position of each group (2,4,6,8,...)
    end_val_gp = np.array([2 * i + 2 for i in range(nbr_group)])
    # Max number of variable allowed in the model
    max_variable = np.copy(nbr_group)
    # limit the maximum number of variables ever to be nonzero. (stopping criterion)
    pmax = int(min(1.2 * max_variable, nbr_group))
    # Length of Lambdas
    nlambda = int(len(Lambdas))

    # Integer whether to include an intercept (1) or not (0)
    intercept = int(intercept)

    # Parameters that the function ls_f returns
    # Sequence of intercept for Lambdas decreasing
    b0 = np.zeros(nlambda)
    # Sequence of beta for Lambdas decreasing
    beta = np.zeros((nvars, nlambda))#, order='F', dtype=float

    #  actual number of lambda values (solutions), Depending on the error flag jerr (see below)
    # Might return only solutions for larger lambdas (1:(k-1)).
    n_computed_lam = np.copy(nlambda) #np.zeros(1, dtype=np.int32)
    # active set for the last lambdas (0=value not in the active set)
    idx = np.zeros(pmax, dtype=np.int32)
    # number of active variable at each lambda
    n_computed_beta = np.zeros(nlambda, dtype=np.int32)
    # lambda values corresponding to each solution
    computed_lam = np.zeros(nlambda)

    # actual number of passes over the data for all lambda values
    npass = int(0) #np.zeros(1, dtype=np.int64)
    # Error flag jerr:
    #            jerr  = 0 => no error
    #            jerr > 0 => fatal error - no output returned
    #                     jerr < 7777 => memory allocation error
    #            jerr < 0 => non fatal error - partial output:
    #                     Solutions for larger lambdas (1:(k-1)) returned.
    #                     jerr = -k => convergence for kth lambda value not reached
    #                            after maxit (see above) iterations.
    #                     jerr = -10000-k => number of non zero coefficients along path
    #                            exceeds pmax (see above) at kth lambda value.
    jerr = int(0)#np.zeros(1, dtype=np.int64)
    # End  -----------------------------------------------------------------
    flmin = 1.0

    #gc.set_debug(gc.DEBUG_SAVEALL)    
    if verbose>=1:    
        print('gglasso: call fortran code')

           
    count= 0
    #res_norm = np.inf
    maxiter = basis_pursuit_maxiter
    res_norms = []
    lambda_try = []
    continue_loop = True
    #resnorm_less_than_tol = 0
    while continue_loop:
        

        glasso_func.ls_f(gp_size, start_val_gp, end_val_gp, 
                         gamma, X, 
                         y,beta_init, weights, max_variable, 
                         flmin, Lambdas, 
                         tol, maxit, intercept, n_computed_lam, 
                         b0, beta, idx, n_computed_beta, 
                         computed_lam, npass, jerr, verbose_fortran,
                         scaley)          
        # Check error flag
        errormessage = "//////////////////////////ERROR MESSAGE///////////////////////////\n"
        if jerr > 0 and jerr < 7777:
            # fatal error - no output returned
            # jerr < 7777 => memory allocation error
            print(errormessage, "Memory allocation error")
            return 0, 0, 0, 0, 0, 0, 0, 0, 0
        if jerr < 0:
            print(errormessage, "Convergence for kth lambda value not reached after ", maxit, " iterations.")
            print("Partial sequence of solutions returned. (for larger lambdas (1:(", -(jerr - 1), ")) returned.)")

    # Collect returned values
    #n_computed_lam = n_computed_lam[0]
        n_computed_beta = n_computed_beta[0:n_computed_lam]
        nbetamax = np.max(n_computed_beta)
        computed_lam = computed_lam[0:n_computed_lam] / scaley
        if nbetamax > 0:  # i.e. the active set is not empty for all lambda
            beta = beta[:, 0:n_computed_lam] / scaley
        else:
            beta = np.zeros((ncol, n_computed_lam))
        beta_out = beta.copy()
        active_set = np.nonzero(idx)[0]
    
        beta0 = meany + b0[0:n_computed_lam] / scaley
        model = beta0[n_computed_lam - 1] + X.dot(beta[:, n_computed_lam - 1][:, np.newaxis])
        
        res_norm = np.linalg.norm(yin - model[:,0])


        res_norms.append(res_norm)
        lambda_try.append(Lambdas[0].copy())
#        print('in gglasso: ------------- ', lambda_try)
        
        continue_loop = res_norm>bp_tol_in and count < maxiter
        
        beta_init = beta_out[:,0]
        count += 1
        Lambdas = Lambdas/2.0
        
    if count >= maxiter:
        print('Warning: in gglasso_basis_pursuit, x such that ||Ax-y||<epsilon could not be found' )

    # dichotomy
    print('Max. number of lambda tuning iterations: {}'.format(maxiter))
    print('Basis pursuit tolerance (epsilon) = {}'.format(bp_tol_in))
    if np.linalg.norm(beta)>0 and count<maxiter:
        count = 0   
        lambda_min = lambda_try[-1]
        lambda_max = lambda_try[-2]
        y1 = res_norms[-1]
        y2 = res_norms[-2]
        a = (y2 - y1)/(lambda_max - lambda_min)
        b = -a * lambda_min + y1
        lambda_c = (bp_tol_in -b)/a  #np.sqrt(lambda_min*lambda_max)
#        print('ingglasso: ', lambda_min, lambda_max, lambda_c, res_norm, bp_tol_in)
        plt.figure()
        while np.abs(res_norm - bp_tol_in) > bp_bound_numerical_tol and count<maxiter:

            glasso_func.ls_f(gp_size, start_val_gp, end_val_gp, 
                             gamma, X, 
                             y,beta_init, weights, max_variable, 
                             flmin, lambda_c, 
                             tol, maxit, intercept, n_computed_lam, 
                             b0, beta, idx, n_computed_beta, 
                             computed_lam, npass, jerr, verbose_fortran,
                             scaley)          
            # Check error flag
            errormessage = "//////////////////////////ERROR MESSAGE///////////////////////////\n"
            if jerr > 0 and jerr < 7777:
                # fatal error - no output returned
                # jerr < 7777 => memory allocation error
                print(errormessage, "Memory allocation error")
                return 0, 0, 0, 0, 0, 0, 0, 0, 0
            if jerr < 0:
                print(errormessage, "Convergence for kth lambda value not reached after ", maxit, " iterations.")
                print("Partial sequence of solutions returned. (for larger lambdas (1:(", -(jerr - 1), ")) returned.)")
    
        # Collect returned values
            n_computed_beta = n_computed_beta[0:n_computed_lam]
            nbetamax = np.max(n_computed_beta)
            computed_lam = computed_lam[0:n_computed_lam] / scaley
            if nbetamax > 0:  # i.e. the active set is not empty for all lambda
                beta_out = beta.copy()
                beta = beta[:, 0:n_computed_lam] / scaley
            else:
                beta = np.zeros((ncol, n_computed_lam))
            active_set = np.nonzero(idx)[0]
        
            beta0 = meany + b0[0:n_computed_lam] / scaley
            model = beta0[n_computed_lam - 1] + X.dot(beta[:, n_computed_lam - 1][:, np.newaxis])
            
            res_norm = np.linalg.norm(yin - model[:,0])
            
            if res_norm>bp_tol_in:
                lambda_max = lambda_c
                y2 = res_norm
            else:
                lambda_min = lambda_c
                y1 = res_norm
            
            a = (y2 - y1)/(lambda_max - lambda_min)
            b = -a * lambda_min + y1
            lambda_c = (bp_tol_in -b)/a          
            #lambda_c = np.sqrt(lambda_min*lambda_max)
            print('lambda_min, lambda_max, lambda_c: {} {} {}'.format(lambda_min, lambda_max, lambda_c))
            print('y1, y2, res_norm                : {} {} {}'.format(y1, y2, res_norm))
            plt.plot([lambda_min, lambda_max, lambda_c], [y1, y2, res_norm],'o')            
#            print('in gglasso: ', np.linalg.norm(yin),  np.linalg.norm(model))
#            print('in gglasso: ', beta0)
#            print('in gglasso: ', res_norm, bp_tol_in)
            count +=1

    varnames = ['active_set', 'beta0', 'beta', 'model',
             'computed_lam']
    variables = [active_set, beta0, beta, model, computed_lam] 
    dict_out = dict(zip(varnames,variables))
    return dict_out


def check_argument(y, X, Lambdas, beta_init, weights, maxit, tol, epsilon, intercept):
    """
    Internal function: Do not call.
    """
    ok = True
    message = ""
    if len(y.shape) > 1:
        message += "y must be a one dimensional array (dim=" + str(len(y.shape)) + ")\n"
        ok = False
    if len(y) != np.size(X, 0):
        message += "Lengths of y and X do not agree (number of rows of X=" + str(
            np.size(X, 0)) + "), length of y=" + str(len(y)) + ")\n"
        ok = False
    if np.size(X, 1) % 2 == 1:
        message += "Odd number of columns of the matrix X (number of columns=" + str(np.size(X, 1)) + ")\n"
        ok = False
    if Lambdas is not None:
        if np.any(Lambdas < 0):
            message += "The sequence of Lambdas must be positive\n"
            ok = False
        if not (isinstance(Lambdas, int) or isinstance(Lambdas, float)):
            if len(Lambdas.shape) > 1:
                message += "Lambdas must be a one dimensional array (dim=" + str(len(Lambdas.shape)) + ")\n"
                ok = False
    if beta_init is not None:
        if len(beta_init.shape) > 1:
            message += "beta_init must be a one dimensional array (dim=" + str(len(beta_init.shape)) + ")\n"
            ok = False
        if len(beta_init) != np.size(X, 1):
            message += "Lengths of beta_init and X do not agree (number of columns of X=" + str(
                np.size(X, 1)) + "), length of beta_init=" + str(len(beta_init)) + ")\n"
            ok = False

    if weights is not None:
        if np.any(weights < 0):
            message += "The weights must be positive\n"
            ok = False

        if len(weights.shape) > 1:
            message += "The weights must be a one dimensional array (dim=" + str(len(weights.shape)) + ")\n"
            ok = False

        if len(weights) != int(np.size(X, 1) / 2):
            message += "The weights do not have the right length (length=" + str(len(weights)) + ")\n"
            ok = False
        if np.max(weights) <= 0:
            message += "All the weights are 0\n"
            ok = False

    if maxit != "default":
        if not isinstance(maxit, int):
            message += "The maximum number of iterations (maxit) must be an integer\n"
            ok = False

    if tol < 0:
        message += "The tolerance (tol) should be positive\n"
        ok = False
    if epsilon < 0:
        message += "Epsilon should be positive\n"
        ok = False
    if int(intercept) not in [0, 1]:
        message += "The intercept should be a boolean\n"
        ok = False
    return ok, message

