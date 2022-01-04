# -*- coding: utf-8 -*-

# Copyright 2020 Nathan Hara
#
#  This file is part of the l1periodogram code.
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

import pandas as pd
import numpy  as np
from l1periodogram import filter_poly


def create_dataset(dataset_names, 
                   outlier_cond = 3.5,
                   sigmas_inst = None, 
                   bintimescale = 0,
                   trenddegree=2):
    
    '''
    Concatenates the data from different files 
    and puts them in 
    chronologic order, puts the data in m/s and
    removes outliers on original and detrended data.
    INPUTS:
        - dataset_names: path to the files. 
          Columns 1, 2, 3 of the files should be
          time (rjd), radial velocity, errror on rv
        - outlier_cond: points deviating from the median by
          more than outlier_cond * MAD/0.67 are removed
          from the data sets (MAD = median absolute deviation).
          Then, we fit a linear trend in the data and repeat the exclusion of points deviating from the median by
          more than outlier_cond * MAD/0.67 on the residuals.
        - sigmas_inst: list of additional jitters per instrument (in m/s)
        - bintimescale: data points taken within a time < bintimescale are binned
        - trenddegree: Outliers are removed beforee and after a polynomial trend of 
        degree trenddegree has been fitted
            
    Outputs:
        - T: times of the combined data sets
        - y: data of the combined data set
        - err: error of the combined data set
        - offsets: matrix (Number of measure times x number of files) 
                   its elements are 0 or 1 the row i, column j element 
                   is equal to 1 if the measurement 
                   at time T[i] is from file j and 0 otherwise
        - list_dict_ancillary_variables:
            list whose elements are dictionary,one 
            per file. The keys are the name of the 
            variables and the values are the list of values
            of the corresponding values in the file in np.array format
          
    '''

    T = np.array([])
    y = np.array([])
    err = np.array([]) 
    dict_allvariables = []
    
    ndataset = len(dataset_names)
    Ndata1 = []
    j = 0
    dataset_names_out = dataset_names.copy()
    
    dict_data_removed = {}
    count = 0
    if sigmas_inst is None:
        sigmas_inst = np.zeros(ndataset)
        
    for name in dataset_names:
        df = pd.read_csv(name, sep='\s+')
        listname = df.iloc[0,:]
        df.columns=listname##
        df.drop(df.index[[0]], inplace=True)
        df.rename(columns=listname)
        #If inputs are strings they are set to NaN
        df = df.apply(pd.to_numeric,errors='coerce')

        
        if len(df)>1: # At least two points in dataset
            T1 =   np.array(df.iloc[:,0])  #Time array
            if bintimescale>0:
                #nt1 = len(T1)
                ind = []
                i=0
                #dfs = df
#                while i < nt1:
#                    c = i+1
#                    while c <nt1 and np.abs(T1[c] - T1[i]) < bintimescale:
#                        c+=1
#                    index = list(np.arange(i,c))
#                    i = c
#                    dfs.append(df.iloc[index].mean(axis=0))
                
                #df = pd.concat(dfs,axis=1).T  
            if len(df)>1:              
                T1 =   np.array(df.iloc[:,0])
                y1 =   np.array(df.iloc[:,1]) #Velocity  data
                err1 = np.array(df.iloc[:,2]) #Errors  data
                    
                #Add error corresponding to the instrument
                err1 = np.sqrt(err1**2 + sigmas_inst[count]**2)                 
                        
                    
                
                #Center data
                y1 -= np.mean(y1)
                
                # ------- removing outliers ------- #
                mediany1 = np.median(y1)
                mad = np.median(np.abs(y1 - mediany1))
                sigma = mad/0.67    
                index_condition = np.abs(y1 - mediany1) <outlier_cond*sigma 
                dict_data_removed['T_removed_init'] = T1[index_condition==False]
                dict_data_removed['y_removed_init'] = y1[index_condition==False]
                dict_data_removed['err_removed_init'] = err1[index_condition==False]
                T1 = T1[index_condition]
                y1 = y1[index_condition]
                err1 = err1[index_condition]  
                tab = df.iloc[index_condition,:].values
                
        
                #remove outliers on detrended data
                const = np.ones((len(T1),1))
                model, theta, M_H00 = filter_poly.filterpoly(T1,y1,np.eye(len(T1)),const,trenddegree)
                detrendy1 = y1 - model
                detrendmediany1 = np.median(detrendy1)
                detrendmad = np.median(np.abs(detrendy1 - detrendmediany1))
                detrensigma = detrendmad/0.67   
                index_condition = np.abs(y1 - mediany1) <outlier_cond*detrensigma
                dict_data_removed['T_removed_afterdrift'] = T1[index_condition==False]
                dict_data_removed['y_removed_afterdrift'] = y1[index_condition==False]
                dict_data_removed['err_removed_afterdrift'] = err1[index_condition==False]
                T1 = T1[index_condition]
                y1 = y1[index_condition]
                err1 = err1[index_condition]
                tab2 = tab[index_condition,:]
                
                # ------- remove outliers end ------- #
            
                #Data binning
                nobs = len(T1)
                nights_labels = np.array(['0']*nobs, dtype=object)
                
                c = 0
                i0 = 0
                for i in range(nobs-1):
                    if np.abs(T1[i+1] - T1[i0])<bintimescale:
                        nights_labels[i+1] = nights_labels[i]
                    else:
                        c +=1
                        nights_labels[i+1] = str(c)
                        i0 = i+1          
            
                err2 = np.array(err1**2)
                nkeys = tab2.shape[1]
                
                #print('nkeys',nkeys)
                #print('l(t1)',len(T1))
                #print('nights_labels',nights_labels[-1])
                #print(tab2[:,1])
                data_binned = tab2.copy()
                for i in range(nkeys):  
                    nightschecked = []   
                    binned_timeseries = []
                    for s in nights_labels:        
                        if s not in nightschecked:       
                            condition = nights_labels==s                            
                            err2s = err2[condition]
                            timeseries = tab2[condition,i]
                            #print(i)
                            try:
                                timeseriesbin = np.sum(timeseries/err2s) / np.sum(1/err2s)
                                binned_timeseries.append(timeseriesbin)
                            except:
                                None   
                            nightschecked.append(s)
                    l = len(binned_timeseries) 
                    #print('l',l)
                    try:
                        data_binned[:l,i] = np.array(binned_timeseries)   
                    except:
                        None   
                    #print('number of points after binning', len(data_binned['']))     
                data_binned = data_binned[:l,:]
                
                T1 = data_binned[:,0]
                y1 = data_binned[:,1]
                err1 = data_binned[:,2]

                if np.mean(err1)<0.05:
                    #transform from km/s to m/s
                    y1 = y1*1000
                    err1 = err1*1000

                Ndata1.append(len(T1))         
                T = np.concatenate((T,T1))
                y = np.concatenate((y, y1))
                err  = np.concatenate((err, err1))
                dict_allvariables.append(dict(zip(listname,data_binned.T))) 
                j = j+1        
        
                
            
            else:
                 ndataset = ndataset - 1 
                 dataset_names_out.remove(name)
        
        else:
            ndataset = ndataset - 1 
            dataset_names_out.remove(name)
    
    Nt = len(T)
    offsets = np.zeros((Nt, ndataset))
    ind1 = 0
    for i in range(ndataset):
        ind2 = ind1 + Ndata1[i]
        offsets[ind1:ind2,i] = 1
        ind1 = ind2
       
    ind = np.argsort(T)
    T = T[ind]
    y = y[ind]
    err = err[ind]    
    offsets = offsets[ind,:]

    
    return(T,y,err,offsets, 
           dict_allvariables, 
           dict_data_removed,
           dataset_names_out)
    
    
    
def smooth_ts(T, y, Tpredict, smoothing_style = None,return_covar=False):
    
    #Style :dictionary with options 
    #    method = 'gp', or 'rkhs'
    #    for 'gp' / 'rkhs' define the time scale tau, the error on the 
    #    observations sigmaWs (size of T) and the variance of 
    if smoothing_style is None:
        return(T,y)
    
    indexdef = [i for (i,v) in enumerate(y) if v is not np.nan]    
    Tdef = T[indexdef]
    ydef = y[indexdef]
       
    
    if smoothing_style['method'] == 'gp'    or \
       smoothing_style['method'] == 'rkhs':

           if smoothing_style['kernel'] == 'gaussian':
               tau = smoothing_style['tau']
               sigmaWs = smoothing_style['sigmaWs']
               sigmaR = smoothing_style['sigmaR']
               k_stars_T = gaussian_kernel(Tpredict,Tdef
                                           ,sigmaR, tau)
               print(k_stars_T)
               
               K = gaussian_kernel(Tdef,Tdef, sigmaR, tau, sigmaWs=sigmaWs)
               alpha = np.linalg.solve(K,ydef)               
               ypredict = k_stars_T.dot(alpha)

               if return_covar:
                   Kpredict = gaussian_kernel(Tpredict,Tpredict,
                                          sigmaR, tau)
                   Wk_stars = np.linalg.solve(K,k_stars_T.T) 
                   covar_predict = Kpredict - k_stars_T.dot(Wk_stars)                
               
    if return_covar:     
        return(ypredict,covar_predict)
    else:
        return(ypredict)
                   
                   

def gaussian_kernel(t1,t2, sigmaR, tau,sigmaWs=None):
    
    tau2 =  tau**2
    N1 = len(t1)
    N2 = len(t2)
    output = np.zeros((N1,N2))
    for i in range(N1):
        for j in range(N2):
            v = -0.5*(t1[i] - t2[j])**2/tau2
            output[i,j] = v
            
    output = sigmaR**2 * np.exp(output)
    
    #!!! Only if the matrix is square
    if sigmaWs is not None:
        output = output + np.diag(sigmaWs**2)
        
    return(output)
        
    
'''
sstyle['method']='rkhs'
sstyle['sigmaWs'] = 0.5*np.std(df['bis_span'])*np.ones(len(df['bis_span']))
sstyle['sigmaR'] = np.std(df['bis_span'])
sstyle['tau'] = 3.0

smbis = combine_datasets.smooth_ts(T, df['bis_span'], Tpredict, smoothing_style =sstyle)

plt.plot(T,df['bis_span'],'o')
plt.plot(Tpredict,smbis)            
'''      
                   
    
    
    
    
    

