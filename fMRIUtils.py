import numpy as np
import matplotlib.pyplot as plt

def jaccardIndex(img1,img2):
    i = np.logical_and(img1,img2)
    u = np.logical_or(img1,img2)
    return(sum(i)/sum(u))

def SNR(sPred,sReal):
    e = sPred-sReal
    return(np.median(sPred)/e.std())

def CNR(sPred,sReal):
    a = max(sPred)-np.median(sPred)
    e = sPred-sReal
    return(a/e.std())

def plot2DImage(imgNPArray,dim,save,fn):
    fig, ax = plt.subplots(figsize=(2,2))
    ax.imshow(imgNPArray,cmap='YlOrRd')
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    if save:
        fnsvg = fn + '.svg'
        fig.savefig(fnsvg)