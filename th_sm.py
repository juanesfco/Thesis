import scipy as sp
import numpy as np
import pandas as pd
import fMRIUtils as fmriu

def th_sm(Xi):
    X = [Xi]
    n = len(Xi)
    Zeta = [np.zeros(n)]
    N = [n]
    eta = []
    k = 0
    t = 0
    while not(t) and k < 20:
        ## 2a)
        # MLE of x^(k-1) -> check a,b=0,1
        A,B,loc,scale = sp.stats.truncnorm.fit(X[k],fa=0,fb=1)
        # Apply gaussian smoothing to obtain x^(k) -> check sigma
        s = np.exp(k/20) # Exponential increase
        X.append(sp.ndimage.gaussian_filter(np.reshape(X[k], (-1, 200)),sigma=s).flatten())
        
        ## 2c)
        # Get a_n_k-1, b_n_k-1 and i -> check i
        rho = 0.95
        bn = rho*sp.stats.truncnorm.ppf(1-1/n,A,B,loc,scale)
        an = rho/(n*sp.stats.truncnorm.pdf(bn/rho,A,B,loc,scale))
        locg,scaleg = sp.stats.gumbel_r.fit(X[k])
        i = sp.stats.gumbel_r.ppf(0.95,locg,scaleg)
        # Calculate eta_k
        eta.append(an*i+bn)
        # Compare
        Zeta.append(X[k+1]>eta[k])
        ni = n-sum(Zeta[k+1])
        if k == 0 and ni == n:
          t = 1
          print('no activation')
          return np.array(Zeta), np.array(N), np.array(eta), np.array(X)
        else:
          N.append(ni)
        ## Calculate JaccardIndexes
        if k>=2:
          J_1 = fmriu.jaccardIndex(Zeta[k-2],Zeta[k-1])
          J_2 = fmriu.jaccardIndex(Zeta[k-1],Zeta[k])
          #print(k,ni,J_1,J_2)
          if J_1 >= J_2:
            t = 1
            print('termination by Jaccard Index')
            Zeta.pop()
            N.pop()
            eta.pop()
            X.pop()
            return np.array(Zeta), np.array(N), np.array(eta), np.array(X)
        
        ## Add iteration
        k += 1
    return np.array(Zeta), np.array(N), np.array(eta), np.array(X)

path = 'Data/Simulations/'

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
