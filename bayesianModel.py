import numpy as np
from scipy.stats import invgamma
import pandas as pd
import os
import sys
import time

# Define Bayesian model function

def betas(X,y):
  V_B = np.linalg.inv(X.T@X)
  Beta_v = (V_B@X.T)@y

  n=len(y)
  k=1 # Number of parameters

  ymXB = y-X@Beta_v
  s_2 = ymXB.T@ymXB/(n-k) 

  v = n-k
  alpha = v/2
  beta = v/2*s_2

  sigma_2 = invgamma.rvs(a=alpha,scale=beta,size=1000)

  betas = []
  for i in range(1000):
    beta = np.random.multivariate_normal(Beta_v[:,0],V_B*sigma_2[i])
    betas.append(beta)

  betas = np.array(betas)

  return(betas)

def main(pmin, pmax, qmin, qmax, R, path = 'Data/Simulations/'):
  fn_X = path + 'X.csv'
  X = pd.read_csv(fn_X)
  # For each P and Q
  for p in range(pmin,pmax):
    for q in range(qmin,qmax):
      # Run model and obtain probability
      pMaps_ar = []
      for r in range(R):
          fn = path + 'BOLD_P' + str(p) + 'Q' + str(q) + 'R' + str(r+1) + '.csv'
          m = True
          while m:
            mm = os.path.isfile(fn)
            if mm:
              print(fn, " found, sleeping 30 seconds.")
              time.sleep(30)
              Y = pd.read_csv(fn)
              probs = []
              for v in Y.columns:
                  print('p:',p,' - q:',q,' - r:',r+1,'v: ', v)
                  b = betas(X.values,Y[v].values[np.newaxis].T)
                  probs.append(sum(b[:,0]>0)/1000)
              pMaps_ar.append(probs)
              m = False
      pMaps = pd.DataFrame(np.array(pMaps_ar))
      # Save dataframe
      fnsave = path + 'pMaps_P' + str(p) + 'Q' + str(q) + '.csv'
      pMaps.to_csv(fnsave,index=False)
  print('Probability Maps Saved')

if __name__ == "__main__":
    main(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]))