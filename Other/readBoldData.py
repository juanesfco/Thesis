import matplotlib.pyplot as plt
import numpy as np
import bayesianModel as bM

def voxelTimeSeries(img,x,y,z):
  data = img.get_fdata()
  
  return(data[x,y,z,:][np.newaxis].T)

def voxelTimeSeries_masked(v,img_masked):
  return(img_masked[:,v][np.newaxis].T)

def slices2D(img,x,y,z,t):
  data = img.get_fdata()

  xslc = data[x,:,:,t]
  yslc = data[:,y,:,t]
  zslc = data[:,:,z,t]

  slices = [xslc, zslc, yslc]

  fig, axes = plt.subplots(1, len(slices), figsize=(15, 45))
  plt.subplots_adjust(wspace=.4)
  for i, slice in enumerate(slices):
    axes[i].imshow(slice.T, cmap="inferno", origin="lower")

def SNR(X,img):
  n = len(img[0])
  #n = 10
  SNR_Image = []
  for i in range(n):
    y = voxelTimeSeries_masked(i,img)
    betas = bM.betas(X.values,y)
    print((i+1)/n*100)

    e = (X@(betas[0,:])).values[np.newaxis].T-y
    SNR_Image.append(y.mean()/np.linalg.norm(e))

  return(np.array(SNR_Image))
