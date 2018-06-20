#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:32:12 2018
https://stackoverflow.com/questions/37424775/python-finding-the-intersection-of-two-gaussian-kde-functions-objects
@author: gonthier
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.neighbors.kde import KernelDensity

# Fit KDE
def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
#    print(kde_skl.score_samples(x[:, np.newaxis]))  # returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return kde_skl, np.exp(log_pdf)

# Find intersection
def findIntersection(fun1, fun2, lower, upper):
    return brentq(lambda x : fun1(x) - fun2(x), lower, upper)

if __name__ == '__main__':

    # Generate normal functions
    x_axis = np.linspace(-3, 3, 100)
    gaussianA = norm.pdf(x_axis, 2, 0.5)  # mean, sigma
    gaussianB = norm.pdf(x_axis, 0.1, 1.5)
    
    # Random-sampling from functions
    a_samples = norm.rvs(2, 0.5, size=100)
    b_samples = norm.rvs(0.1, 1.5, size=100)
    
    kdeA, pdfA = kde_sklearn(a_samples, x_axis, bandwidth=0.25)
    kdeB, pdfB = kde_sklearn(b_samples, x_axis, bandwidth=0.25)
    
    funcA = lambda x: np.exp(kdeA.score_samples([x][0]))
    funcB = lambda x: np.exp(kdeB.score_samples([x][0]))
    
    result = findIntersection(funcA, funcB, -3,3)
    
    # Plot
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(x_axis, gaussianA, color='green')
    ax1.plot(x_axis, gaussianB, color='blue')
    ax1.set_title('Original Gaussians')
    ax2.plot(x_axis, pdfA, color='green')
    ax2.plot(x_axis, pdfB, color='blue')
    ax2.set_title('KDEs of subsampled Gaussians')
    ax2.axvline(result, color='red')
    plt.show()