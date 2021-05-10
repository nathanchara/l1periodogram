# l1 periodogram

This repository contains a Python implementation of the l1 periodogram as 
described in Hara, Boué, Laskar, Correia 2017, MNRAS, Vol. 464, Issue 1, p.1220-1246, with a few more features.
please cite this paper if you are using the code. 

### Description

The l1 periodogram is designed to search for periodicities in unevenly sampled time series. 
It can be used similarly as a Lomb-Scargle periodogram, 
and retrieves a figure which has a similar aspect but has fewer peaks due to aliasing. 
It is primarily designed for the search of exoplanets in radial velocity data, but can be also used for other purposes. 

The principle of the algorithm is to search for a representation of the input signal 
as a sum of a small number of sinusoidal components, that is a representation which is sparse in the frequency domain. 
Here, "small number" means small compared to the number of observations. 

The code is based on the Basis Pursuit minimization problem (Chen & Donoho 1998). 
In the present code, the Basis Pursuit problem can be solved with 
- The LARS algorithm (Efron, Hastie, Johnston, Tibshirani, R. 2004, Ann. Statist., Volume 32, Number 2, 407-499.)
- The gglasso algorithm (Yang & Zou 2014. Statistics and Computing. 25(6), 1129-1141.)
To use the latter, run the following command in the l1periodogram_codes folder. 
```bash
python -m numpy.f2py -c gglasso.f90 -m gglasso_wrapper
```
When using a notebook, restart the kernel after running this command.

### Necessary inputs

The mandatory inputs are: 
- A time series
- The epochs of observation of the time series
- A covariance model for the noise.

The other parameters are set by default, the most critical ones are:
- The maximum frequency of the frequency grid
- The unpenalized vectors: linear predictors that are known to be in the data (like offsets, trends, activity models). 

### Getting started

Download the l1periodogram repository onto your local computer. The notebook l1_periodogram_tutorial_I.pynb will walk you through the different features of the code. The notebook l1_periodogram_tutorial_II.pynb gets deeper into the details of what the code does. 

You will also find a notebook dedicated to reproducing the results of Hara, Bouchy, Stalport, Boisse et al. 2020, A&A, 636, L6, called l1_periodogram_HD158259_analysis.ipynb


### Credits
Written by Nathan C. Hara, with contributions from Alessandro R. Mari
