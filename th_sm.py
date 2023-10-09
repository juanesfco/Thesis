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
import sys
import os

def th_sm(Xi,dim):
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
  S = np.arange(0.75,3.50,0.25)
  while not(t) and k < 9: # Loop depends on iteration number and flag
      ## Step 2. Fit X[k] to a truncated normal distribution between 0 and 1
      #A,B,loc,scale = sp.stats.truncnorm.fit(X[k],fa=0,fb=1)
      A,B,mu,sigma = 0,1,X[k].mean(),X[k].std()

      alpha,beta = (A - mu)/sigma, (B - mu)/sigma
      Z = sp.stats.norm.cdf(beta) - sp.stats.norm.cdf(alpha)
      phi_alpha = sp.stats.norm.pdf(alpha)
      phi_beta = sp.stats.norm.pdf(beta)

      loc = mu+((phi_alpha-phi_beta)/Z)*sigma 
      var = (sigma**2)*(1-((beta*phi_beta-alpha*phi_alpha)/Z)-((phi_alpha-phi_beta)/Z)**2)
      scale = var**(0.5)
      #print('mu: ',mu, 'sigma: ', sigma)
      #print('loc: ',loc, 'scale: ', scale)
      
      ## Step 3. Apply gaussian smoothing to obtain X[k+1]
      s = S[k] # Value of sigma increases with iterations
      # The images are reshaped for the gaussian filter and then flattened agai.
      X.append(sp.ndimage.gaussian_filter(np.reshape(X[k], (-1, dim)),sigma=s).flatten())

      ## Step 4. Calculate threeshold
      # Get bn - Using values obtained in Step 2
      bn = sp.stats.truncnorm.ppf(1-1/n,A,B,loc,scale)
      # Get an - Using values obtained in Step 2
      an = 1/(n*sp.stats.truncnorm.pdf(bn,A,B,loc,scale))
      # Get i - Fitting image to Gumbel distribution and taking upper tail value
      locg,scaleg = sp.stats.gumbel_r.fit(X[k+1])
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
          print('Termination by Jaccard Index in ', k, ' iterations.')
          # Eliminate the last values of the list because the Jaccard Index descreased.
          Zeta.pop()
          N.pop()
          eta.pop()
          X.pop()
          return np.array(Zeta), np.array(N), np.array(eta), np.array(X)
      # Increase iteration marker
      k += 1
  return np.array(Zeta), np.array(N), np.array(eta), np.array(X)

def main(imName, dim, pmin, pmax, qmin, qmax, R ,path = 'Data/Simulations/'):
  fn_ball = path + imName + '_' + str(dim) + '.npy'
  b = np.load(fn_ball)
  for p in range(pmin,pmax): # Change to 4
      for q in range(qmin,qmax): # Change to 4
          fn = path + 'pMaps_P' + str(p) + 'Q' + str(q) + '.csv'
          m = True
          while m:
            mm = os.path.isfile(fn)
            if mm:
              df = pd.read_csv(fn)
              for r in range(R): # Change to 50
                  X0 = df.loc[r,:].values
                  print("For P,Q,R: ",p,q,r+1)
                  Zeta, N, eta, X = th_sm(X0,dim)
                  print("Final Jaccard Index wrt Original Ball: ",fmriu.jaccardIndex(Zeta[len(Zeta)-1],b))
                  fnZeta = path + 'Zeta_P' + str(p) + 'Q' + str(q) + 'R' + str(r+1) + '.npy'
                  fnN = path + 'N_P' + str(p) + 'Q' + str(q) + 'R' + str(r+1) + '.npy'
                  fneta = path + 'eta_P' + str(p) + 'Q' + str(q) + 'R' + str(r+1) + '.npy'
                  fnX = path + 'X_P' + str(p) + 'Q' + str(q) + 'R' + str(r+1) + '.npy'
                  np.save(fnZeta,Zeta)
                  np.save(fnN,N)
                  np.save(fneta,eta)
                  np.save(fnX,X)
              m = False

if __name__ == "__main__":
    main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6]),int(sys.argv[7]))