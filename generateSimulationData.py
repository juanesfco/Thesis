import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from nilearn.glm.first_level import make_first_level_design_matrix
from statsmodels.tsa.arima.model import ARIMA as ARIMA
from PIL import Image
import sys

# Function to Generate Signal with Noise

def generateNoise(Y,p,q):
  Y_Noise = []
  exog = 0 # media
  PS = [0.5, 0.3, 0.1]
  QS = [0.5, 0.3, 0.1]
  P = PS[:p]
  Q = QS[:q]
  var = 25**2 # varianza
  PAR = [exog] + P + Q + [var]
  N = len(Y[0])

  for vox in range(len(Y)):
    M = ARIMA(Y[vox],order=(p,0,q))
    
    e = M.simulate(PAR,N)
    e = e[np.newaxis].T

    Y_Noise.append(Y[vox] + e)
  
  return(pd.DataFrame((np.array(Y_Noise)[:,:,0]).T))

def main(imName, dim, pmin, pmax, qmin, qmax, R, imtype, path = 'Data/Simulations/'):
  # Generate Simulated Experiment Events and Design Matrix
  rt = 2 # repetition time
  n = 100 # number of scans
  onset = np.arange(n) * rt
  duration = np.ones(100)*2
  trial_type = np.ones(100)*np.nan

  st = [random.randint(5,12)]
  for i in range(3):
    sti = st[i] + random.randint(18,25)
    st.append(sti)

  for s in st:
    duration[s] = 10
    trial_type[s] = 1

  events = pd.DataFrame({'trial_type':trial_type, 'onset':onset, 'duration': duration})
  events.replace(1,'st',inplace=True)

  X = make_first_level_design_matrix(onset, events, drift_model=None)

  # Load Ball Image
  if imtype == 'png':
    fn_img = path + imName + '.png'
    img_open = Image.open(fn_img)
    img_resize = img_open.resize((dim,dim))
    img = np.array(img_resize)

    img_activated = np.amax(img[:,:,0:3],axis=2)<128
    img_mask = img[:,:,3]==255
    img_final = np.logical_and(img_activated,img_mask).flatten()
    
    fn_ball = path + imName + '_' + str(dim) + '.npy'
    np.save(fn_ball,img_final)
  elif imtype == 'npy':
    fn_img = path + imName + '.npy'
    img_final = np.load(fn_img).flatten()
  else:
    print('Image type not found')

  # Create Coefficients Array

  V = len(img_final)
  Betas = []
  for p in img_final:
    if p:
      B = np.array([[75],[100]]) # Can change
    else:
      B = np.array([[0],[100]])

    Betas.append(B)

  # BOLD Signal without Noise

  y = []
  for b in Betas:
    y.append(X.values@b)

  # Save Design Matrix and BOLD without Noise

  fn_X = path + 'X.csv'
  X.to_csv(fn_X,index=False)

  print('X saved')

  BOLDs = pd.DataFrame((np.array(y)[:,:,0]).T)
  fn_BOLD = path + 'BOLD.csv'
  BOLDs.to_csv(fn_BOLD,index=False)

  print('BOLD saved')

  # Generate and save all signals

  for p in range(pmin,pmax+1): # p within [0,1,2,3]
      for q in range(qmin,qmax+1): # q within [0,1,2,3]
          for r in range(R): # number of runs
              print('p:',p,' - q:',q,' - r:',r+1)
              df_BOLD = generateNoise(y,p,q)
              fn_BOLDPQR = path + 'BOLD_P' + str(p) + 'Q' + str(q) + 'R' + str(r+1) + '.csv'
              df_BOLD.to_csv(fn_BOLDPQR,index=False)
  print('BOLD with Noise Saved')

if __name__ == "__main__":
    main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6]),int(sys.argv[7]),sys.argv[8],sys.argv[9])