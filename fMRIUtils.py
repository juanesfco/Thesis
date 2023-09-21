import numpy as np
import matplotlib.pyplot as plt

def jaccardIndex(img1,img2):
    i = np.logical_and(img1,img2)
    u = np.logical_or(img1,img2)
    return(sum(i)/sum(u))

def SNR(sPred,sReal):
    e = sPred-sReal
    return(np.median(sPred)/np.linalg.norm(e))

def CNR(sPred,sReal):
    a = max(sPred)-np.median(sPred)
    e = sPred-sReal
    return(a/np.linalg.norm(e))

def plotImage(img,dim,colorbar = 1):
    plt.imshow(np.reshape(img,(-1,dim)))
    if colorbar:
        plt.colorbar()