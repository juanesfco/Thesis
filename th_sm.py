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
from scipy import stats
import numpy as np
import pandas as pd
import fMRIUtils as fmriu
import sys
import os
import time
from nilearn.masking import unmask
from nilearn.masking import apply_mask
from nilearn.image import new_img_like

def th_sm(Xi,dimx,dimy,dimz=0):
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
  S = np.linspace(0.65,100,10)
  while not(t) and k < 10: # Loop depends on iteration number and flag
      
      ## Step 2a. Apply gaussian smoothing to obtain X[k+1]
      s = S[k] # Value of sigma increases with iterations
      # The images are reshaped for the gaussian filter and then flattened agai.
      if dimy == -1:
        UM = unmask(X[k],dimx)
        Xk_UM = UM.get_fdata()[:,:,:]
        GF = sp.ndimage.gaussian_filter(Xk_UM,sigma=s)
        NIFTI = new_img_like(UM,GF)
        X.append(apply_mask(NIFTI,dimx))
      else:
        if dimz == 0:
          X.append(sp.ndimage.gaussian_filter(X[k].reshape(dimx, dimy),sigma=s).flatten())
        else:
          X.append(sp.ndimage.gaussian_filter(X[k].reshape(dimx,dimy,dimz),sigma=s).flatten())
      
      ## Step 2bi. Fit X[k] to a truncated normal distribution between 0 and 1
      A,B,mu,sigma = 0,1,X[k].mean(),X[k].std()

      alpha,beta = (A - mu)/sigma, (B - mu)/sigma
      Z = sp.stats.norm.cdf(beta) - sp.stats.norm.cdf(alpha)
      phi_alpha = sp.stats.norm.pdf(alpha)
      phi_beta = sp.stats.norm.pdf(beta)

      loc = mu+((phi_alpha-phi_beta)/Z)*sigma 
      var = (sigma**2)*(1-((beta*phi_beta-alpha*phi_alpha)/Z)-((phi_alpha-phi_beta)/Z)**2)
      scale = var**(0.5)
      
      ## Step 2bii. Calculate av and bv
      # Get bv - Using values obtained in Step 2
      bv = sp.stats.truncnorm.ppf(1-1/n,A,B,loc,scale)
      # Get av - Using values obtained in Step 2
      av = 1/(n*sp.stats.truncnorm.pdf(bv,A,B,loc,scale))
      ## Step 2biii. Calculate threeshold
      # Get i - Taking 0.01-upper tail value of standard Gumbel
      i = sp.stats.gumbel_r.ppf(0.99)
      # Append threeshold to eta array
      eta.append(av*i+bv)

      ## Step 2c. Generate activation maps
      # If voxel is activated, it stays activated (logical or)
      # and if probability values are greater than threeshold,
      # voxels get activated.
      Zeta.append(np.logical_or(X[k+1]>eta[k],Zeta[k]))

      ## Step 3. Termination
      # a. If no activation detected in first iteration, terminate.
      ni = n-sum(Zeta[k+1]) # Number of inactive voxels
      if k == 0 and ni == n:
        t = 1
        print('no activation')
        return np.array(Zeta), np.array(N), np.array(eta), np.array(X)
      else:
        # If activation is detected, append the number of inactive voxels to N
        N.append(ni) 
      # b. If Jaccard Index decreases in two successive iterations, terminate.
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

def main(imName, pmin, pmax, qmin, qmax, R,path,dimx,dimy,dimz=0):
  if imName != 'None':
    fn_im = path + imName + '.npy'
    im = np.load(fn_im)
  for p in range(pmin,pmax+1):
      for q in range(qmin,qmax+1):
          fn = path + 'pMaps_P' + str(p) + 'Q' + str(q) + '.csv'
          m = True
          while m:
            mm = os.path.isfile(fn)
            if mm:
              #print(fn, " found, sleeping 30 seconds.")
              #time.sleep(30)
              df = pd.read_csv(fn)
              for r in range(R): # Change to 50
                X0 = df.loc[r,:].values
                print("For P,Q,R: ",p,q,r+1)
                Zeta, N, eta, X = th_sm(X0,dimx,dimy,dimz)
                if imName != 'None':
                  print("Final Jaccard Index wrt Original Ball: ",fmriu.jaccardIndex(Zeta[len(Zeta)-1],im.flatten()))
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
  if len(sys.argv)>10:
    main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6]),sys.argv[7],int(sys.argv[8]),int(sys.argv[9]),int(sys.argv[10]))
  else:
    main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6]),sys.argv[7],int(sys.argv[8]),int(sys.argv[9]))