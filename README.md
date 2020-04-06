# l1 periodogram

This repository contains a Python implementation of the l1 periodogram as 
described in Hara, Boué, Laskar, Correia 2017, MNRAS, Vol. 464, Issue 1, p.1220-1246, 
please cite this paper if you are using the code. 

### Description

The l1 periodogram is designed to search for periodicities in unevenly sampled time series. 
It can be used similarly as a Lomb-Scargle periodogram, 
and retrieves a figure which has a similar aspect but has fewer peaks due to aliasing. 
It is primarily designed for the search of exoplanets in radial velocity data, but can be also used for other purposes. 

The principle of the algorithm is to search for a representation of the input signal 
as a sum of a small number of sinusoidal components, or a sparse reprentation in frequency. 
Here small number means small compared to the number of observations. 

The code is based on the Basis Pursuit minimization problem (Chen & Donoho 1998). 
In the present code, the Basis Pursuit problem can be solved with 
- The LARS algorithm (Efron, Hastie, Johnston, Tibshirani, R. 2004, Ann. Statist., Volume 32, Number 2, 407-499.)
- The gglasso algorithm (Yang & Zou 2015. Statistics and Computing. 25(6), 1129-1141.)
To use the latter, run the following command in the directory where the files of the present distributions are copied.
```bash
python -m numpy.f2py -c gglasso.f90 -m gglasso_wrapper
```

### Necessary inputs

The mandatory inputs are: 
- A time series
- The epochs of observation of the time series
- A covariance model for the noise. If you want to use the nominal uncertainties, use 

The other parameters are set by default, the most critical ones are:
- The maximum frequency of the frequency grid
- The unpenalized vectors: linear predictors that are known to be in the data (like offsets, trends, activity models). 

### Tutorial

The notebook l1_periodogram_tutorial_I will walk you through the different features of the code
