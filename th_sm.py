"""Threesholding and Smoothing Script

This script allows the user to generate fMRI activation maps out of 
probability maps. The program uses adaptive smoothing and threesholding 
in order to estimate as precisely as possible. The images have to be 
squared.

This tool accepts comma separated value files (.csv).

This script requires that `pandas`, `numpy`, `scipy` and `fMRIUtils` be 
installed within the Python environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * th_sm - takes probability map and returns activation map
    * main - loads probability maps from file system and saves 
             simulation results in .npy format
"""

import scipy as sp
import numpy as np
import pandas as pd
import fMRIUtils as fmriu

def th_sm(Xi,dim=200):
  """Adaptive smoothing and threesholding algorithm to estimate 
    activation maps.

    Parameters
    ----------
    Xi : array (1D)
        The initial probability map, flattened to 1 dimension.
    dim : integer
        Dimensions of the square image (dim x dim)
    
    Returns
    -------
    Zeta : list
        a list of arrays (1D) representing activation maps in
        each iteration
    N : list
        a list of integers representing the number of inactive
        voxels in each iteration
    eta : list
        a list of floats representing the threesholds in each
        iteration
    X : list
        a list of arrays (1D) representing probability maps in
        each iteration
    """

  ## Step 1. Initialization
  X = [Xi] # Initialization of X with Xi
  n = len(Xi) # Number of voxels to consider
  Zeta = [np.zeros(n)] # Initialize Zeta with an array of n zeros
  N = [n] # Initialize N with n (all voxels inactive)
  eta = [] # Initialize eta as an empty array
  k = 0 # Iteration marker
  t = 0 # Termination flag
  while not(t) and k < 20: # Loop depends on iteration number and flag
      ## Step 2. MLE of X[k] *** Can change to find scale using average of st dev of image ***
      A,B,loc,scale = sp.stats.truncnorm.fit(X[k],fa=0,fb=1)
      
      ## Step 3. Apply gaussian smoothing to obtain X[k+1]
      s = np.exp(k/20) # Value of sigma increases exponentially with iterations
      # The images are reshaped for the gaussian filter and then flattened agai.
      X.append(sp.ndimage.gaussian_filter(np.reshape(X[k], (-1, dim)),sigma=s).flatten())
      
      ## Step 4. Calculate threeshold
      # Get rho
      rho = 0.95 # *** Change: Calculate using TRF of first row ***
      # Get bn - Using values obtained in Step 2
      bn = rho*sp.stats.truncnorm.ppf(1-1/n,A,B,loc,scale)
      # Get an - Using values obtained in Step 2
      an = rho/(n*sp.stats.truncnorm.pdf(bn/rho,A,B,loc,scale))
      # Get i - Fitting image to Gumbel distribution and taking upper tail value
      locg,scaleg = sp.stats.gumbel_r.fit(X[k])
      i = sp.stats.gumbel_r.ppf(0.95,locg,scaleg)
      # Append threeshold to eta array
      eta.append(an*i+bn)

      ## Step 5. Generate activation maps
      # If voxel is activated, it stays activated (logical or)
      # and if probability values are greater than threeshold,
      # voxels get activated.
      Zeta.append(np.logical_or(X[k+1]>eta[k],Zeta[k]))

      ## Step 6. Termination
      # A. If no activation detected in first iteration, terminate.
      ni = n-sum(Zeta[k+1]) # Number of inactive voxels
      if k == 0 and ni == n:
        t = 1
        print('no activation')
        return np.array(Zeta), np.array(N), np.array(eta), np.array(X)
      else:
        # If activation is detected, append the number of inactive voxels to N
        N.append(ni) 
      # B. If Jaccard Index decreases in two successive iterations, terminate.
      if k>=2:
        J_1 = fmriu.jaccardIndex(Zeta[k-2],Zeta[k-1])
        J_2 = fmriu.jaccardIndex(Zeta[k-1],Zeta[k])
        if J_1 >= J_2:
          t = 1
          print('termination by Jaccard Index')
          # Eliminate the last values of the list because the Jaccard Index descreased.
          Zeta.pop()
          N.pop()
          eta.pop()
          X.pop()
          return np.array(Zeta), np.array(N), np.array(eta), np.array(X)
      # Increase iteration marker
      k += 1
  return np.array(Zeta), np.array(N), np.array(eta), np.array(X)

def main(path = 'Data/Simulations/'):
  for p in range(1): # Change to 4
      for q in range(1): # Change to 4
          fn = path + 'stProb_P' + str(p) + 'Q' + str(q) + '.csv' #change name
          df = pd.read_csv(fn,index_col=0)
          for i in range(2): # Change to 50
              X0 = df.loc[i,:].values
              Zeta, N, eta, X = th_sm(X0)
              fnZeta = path + 'Zeta_P' + str(p) + 'Q' + str(q) + 'R' + str(i+1) + '.npy'
              fnN = path + 'N_P' + str(p) + 'Q' + str(q) + 'R' + str(i+1) + '.npy'
              fneta = path + 'eta_P' + str(p) + 'Q' + str(q) + 'R' + str(i+1) + '.npy'
              fnX = path + 'X_P' + str(p) + 'Q' + str(q) + 'R' + str(i+1) + '.npy'
              np.save(fnZeta,Zeta)
              np.save(fnN,N)
              np.save(fneta,eta)
              np.save(fnX,X)

if __name__ == "__main__":
    main()