import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from nilearn.glm.first_level import make_first_level_design_matrix
from statsmodels.tsa.arima.model import ARIMA as ARIMA

# Path to data

path = './Data/'

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

imgfn = path + 'ball.png'
dim_im = 200

img = plt.imread(imgfn)

nimg = np.ones((dim_im,dim_im))
for i in range(dim_im):
  for j in range(dim_im):
    a = img[i,j,3] 
    if a == 1:
      r,g,b = img[i,j,0], img[i,j,1], img[i,j,2]
      nimg[i,j] = max(r,g,b)

fnimg = (nimg-1)*-1

ball = (fnimg>0.5).flatten()
ballfn = path + 'ball.npy'
np.save(ballfn,ball)

# Create Coefficients Array

ffnimg = fnimg.flatten()
v = len(ffnimg)
Betas = []
for i in range(v):
  if ffnimg[i] > 0.5:
    B = np.array([[75],[100]]) # Can change
  else:
    B = np.array([[0],[100]])

  Betas.append(B)

# BOLD Signal without Noise

y = []
for i in range(v):
  y.append(X.values@Betas[i])

# Function to Generate Signal with Noise

def generateSignal(p,q,i,n):
  M = ARIMA(y[i],order=(p,0,q))

  exog = 0 # media

  PS = [0.5, 0.3, 0.1]
  QS = [0.5, 0.3, 0.1]

  P = PS[:p]
  Q = QS[:q]
  var = 20 # varianza

  PAR = [exog] + P + Q + [var]

  e = M.simulate(PAR,n)
  e = e[np.newaxis].T

  return(y[i] + e)

# Save Design Matrix and BOLD without Noise

X_fn = path + 'Simulations/X.csv'
X.to_csv(X_fn,index=False)

print('X saved')

BOLDs = pd.DataFrame()
for i in range(v):
    cn = 'v' + str(i+1)
    BOLDs[cn] = y[i][:,0]
BOLD_fn = path + 'Simulations/BOLD.csv'
BOLDs.to_csv(BOLD_fn,index=False)

print('BOLD saved')

# Generate and save all signals

for p in range(4): # p within [0,1,2,3]
    for q in range(4): # q within [0,1,2,3]
        #print('p:',p,' - q:',q)
        for run in range(2): # number of runs
            BOLDs = pd.DataFrame()
            for i in range(v):
                print('p:',p,' - q:',q,' - r:',run+1,' - v:',i)
                BOLD = generateSignal(p,q,i,n)
                cn = 'v'+ str(i+1)
                BOLDs[cn] = BOLD[:,0]
            BOLDPQR_fn = path + 'Simulations/BOLD_P' + str(p) + 'Q' + str(q) + 'R' + str(run+1) + '.csv'
            BOLDs.to_csv(BOLDPQR_fn,index=False)