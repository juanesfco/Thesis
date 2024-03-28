import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

ri = 235/256
gi = 235/256
bi = 235/256
ai = 128/256

rf = 128/256
gf = 0/256
bf = 38/256
af = 255/256

ci = [ri,gi,bi,ai]
cf = [rf,gf,bf,af]

cmap = np.ones((192,4))

for i in range(4):
    cmap[:,i] = np.linspace(ci[i],cf[i],192)

cmap = mpl.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])

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

def plot2DImage(imgNPArray,save,fn=''):
    fig = plt.figure(figsize=(2,2))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    fig.add_axes(ax)
    ax.imshow(imgNPArray,cmap=cmap)
    if save:
        fig.savefig(fn,dpi=1000)

def plot3DImage(imgNPArray,save,fn=''):
    fig, ax = plt.subplots(figsize=(2.9,2.9),subplot_kw={"projection": "3d"})

    ax.voxels(imgNPArray,facecolors=[127/256,1/256,38/256,255/256],edgecolors=[127/256,1/256,38/256,255/256])
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.axes.get_zaxis().set_ticks([])

    bbox = fig.bbox_inches.from_bounds(0.52, 0.42, 2, 2)

    if save:
        fig.savefig(fn, transparent=True, bbox_inches = bbox,dpi=1000)