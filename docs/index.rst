l1 periodogram
==============

This repository contains a Python implementation of the l1 periodogram
as described in :footcite:`Hara2017`, with a few more features.
Please cite this paper if you are using the code.

.. footbibliography::
   
Description
~~~~~~~~~~~

The l1 periodogram is designed to search for periodicities in unevenly
sampled time series. It can be used similarly as a Lomb-Scargle
periodogram, and retrieves a figure which has a similar aspect but has
fewer peaks due to aliasing. It is primarily designed for the search of
exoplanets in radial velocity data, but can be also used for other
purposes.

The principle of the algorithm is to search for a representation of the
input signal as a sum of a small number of sinusoidal components, that
is a representation which is sparse in the frequency domain. Here,
“small number” means small compared to the number of observations.

The code is based on the Basis Pursuit minimization problem (:cite:t:`Chen1998`).
In the present code, the Basis Pursuit problem can be solved with

- The LARS algorithm (:cite:t:`Efron2004`)

- The gglasso algorithm (:cite:t:`Yang2015`)

Necessary inputs
~~~~~~~~~~~~~~~~

The mandatory inputs are: - A time series - The epochs of observation of
the time series - A covariance model for the noise.

The other parameters are set by default, the most critical ones are: -
The maximum frequency of the frequency grid - The unpenalized vectors:
linear predictors that are known to be in the data (like offsets,
trends, activity models).

Getting started
~~~~~~~~~~~~~~~

Download the l1periodogram repository onto your local computer. The
notebook ``l1_periodogram_tutorial_I.pynb`` will walk you through the
different features of the code. The notebook
``l1_periodogram_tutorial_II.pynb`` gets deeper into the details of what
the code does.

You will also find a notebook dedicated to reproducing the results of :cite:t:`Hara2020`, called
``l1_periodogram_HD158259_analysis.ipynb``

Credits
~~~~~~~

Written by Nathan C. Hara, with contributions from Alessandro R. Mari

Python packaging by Denis Rosset.

Changelog
~~~~~~~~~

.. include:: ../HISTORY.rst

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: General information

   install
   api
   bibliography

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Examples

   notebooks/l1_periodogram_tutorial_I.ipynb
   notebooks/l1_periodogram_tutorial_II.ipynb
   notebooks/l1_periodogram_HD158259_analysis.ipynb

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: For contributors

   
