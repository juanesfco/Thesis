import numpy as np
import matplotlib.pyplot as plt
import sys

def createActivationCluster(array,i,j,k,size):
  r = int(np.round(np.sqrt(size*3/4/np.pi)))
  array[i,j,k] = 1
  #print('From cluster centered in: ',i,', ',j,', ',k)
  for x in range(-r,r+1):
    for y in range(-r,r+1):
      for z in range(-r,r+1):
        if np.linalg.norm([x,y,z]) <= r:
          try:
            array[i+x,j+y,k+z] = int(np.random.randint(3)>0)
            #print('Activated voxel: ',i+x,', ',j+y,', ',k+z)
          except:
            pass
  #print('Porcentaje activado con clusters: ',sum(sum(sum(array)))/40000)
  activateLoners(array)
  #print('Porcentaje activado final: ',sum(sum(sum(array)))/40000)

def activateLoners(array):
  s = array.shape
  for x in range(s[0]):
    for y in range(s[1]):
      for z in range(s[2]):
        if array[x,y,z] == 0:
          if surrAct(array,x,y,z):
            array[x,y,z] = 1

def surrAct(array,i,j,k):
  s = 0
  for x in [-1,1]:
    for y in [-1,1]:
      for z in [-1,1]:
        try:
          s += array[i+x,j+y,k+z]
        except:
          pass
  if s >= 4:
    return(True)
  else:
    return(False)

def createImage(dimX,dimY,dimZ):
  p = (np.random.random((dimX,dimY))>0.99).astype(int)
  #print('Porcentaje inicial de zonas activadas: ',sum(sum(p))/1600)
  base = np.zeros((dimX,dimY,dimZ))
  s = p.shape
  for i in range(s[0]):
    for j in range(s[1]):
      if p[i,j]:
        k = np.random.randint(0,dimZ)
        createActivationCluster(base,i,j,k,2*dimZ)
  return(base)

if __name__ == "__main__":
    im  = createImage(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
    fn = 'Data/ImagesCreated/image_' + sys.argv[1] + 'x' + sys.argv[2] + 'x' + sys.argv[3] + '_' + sys.argv[4]
    np.save(fn,im)

