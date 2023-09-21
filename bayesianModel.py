import numpy as np
from scipy.stats import invgamma
import pandas as pd

# Import design matrix and one example of signal

X = pd.read_csv('Data/Simulations/X.csv')
Y = pd.read_csv('Data/Simulations/BOLD_P0Q0R1.csv')

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

# Column labels
C = Y.columns

# For each P and Q
for p in range(2):
  for q in range(2):
    # Run model and obtain probability
    st_prob = pd.DataFrame(columns=C)
    for i in range(1):
        fn = 'Data/Simulations/BOLD_P' + str(p) + 'Q' + str(q) + 'R' + str(i+1) + '.csv'
        Y = pd.read_csv(fn)
        probs = []
        for c in C:
            print(p,q,c)
            b = betas(X.values,Y[c].values[np.newaxis].T)
            probs.append(sum(b[:,0]>0)/1000)
        st_prob.loc[len(st_prob.index)] = probs 

    # Save dataframe
    fnsave = 'Data/Simulations/pMap_P' + str(p) + 'Q' + str(q) + '.csv'
    st_prob.to_csv(fnsave)