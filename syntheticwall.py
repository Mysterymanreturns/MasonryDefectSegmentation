#syntheticwall = creates a synthetic masonry wall with random block dimensions and geometry

#INPUTS
#dimheight
#dimwidth
#blockheight
#blockwidth
#saveloc

#OUTPUTS
#wall
#mask


def syntheticwall(dimheight,dimwidth,blockheight,blockwidth,saveloc):
    import random
    import os, sys
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import math
    from PIL import Image, ImageOps
    import noise
    
   # dist = picnp[18:30,10]
    mean = 614400#dist.mean()
    mu = 64
    variance = 1
    width = 50
    sigma = math.sqrt(variance)
    x = np.linspace(mu - width*sigma, mu + width*sigma, 128)
    data =stats.norm.pdf(x, mu, sigma)
    data = data/data.max()
    test = np.zeros([128,128])
    test = test + data
    testcrop = np.transpose(test[0,44:84])
    
    wall = np.zeros([dimwidth,dimheight])
    vspacing = blockheight
    hspacing = blockwidth
    verticaljointoffsetdownwards = random.randint(int(blockwidth/4),int(3*blockwidth/4))
    jointcropsize = testcrop.shape
    jointcropwidth = jointcropsize[0]
    jointcroplength = 1   #jointcropsize[1]
    wallsize = wall.shape
    wallheight = wallsize[0]
    walllength = wallsize[1]
    
    #place horizontal joints
    nhjoints = int(wallheight/vspacing)

    hjoints = np.zeros([wallheight])#jointcroplength])
    startpoint = round(vspacing/2)
    for n in range(nhjoints):
      location = n*vspacing+ startpoint
      start = max(location-int(jointcropwidth/2),0)
      differencestart = start-(location-int(jointcropwidth/2))
      end = min(location+int(jointcropwidth/2),wallheight)
      differenceend = (location+int(jointcropwidth/2))-end
      hjoints[start:end]=np.maximum(hjoints[start:end],testcrop[differencestart:jointcropwidth-differenceend])
    wall = wall+ np.transpose([hjoints])
    
    
    #place vertical joints
    nvjoints = int(walllength/hspacing)


    for n1 in range(nhjoints+1):
      jointoffset = verticaljointoffsetdownwards*(n1)-hspacing*int((verticaljointoffsetdownwards*(n1))/hspacing)
      #print(f"n1 = {n1}")
      for n2 in range(nvjoints+1):
        #print(f"n2 ={n2}")
        location = n2*hspacing+ jointoffset
        start = max(location-int(jointcropwidth/2),0)
        differencestart = start-(location-int(jointcropwidth/2))
        end = min(location+int(jointcropwidth/2),walllength)
        differenceend = (location+int(jointcropwidth/2))-end
        for l in range(vspacing):
          if l+((n1-1)*vspacing+startpoint+1)<=wallheight and l+((n1-1)*vspacing+startpoint+1)>=0 and start <=walllength and end >=0:
            wall[l+((n1-1)*vspacing+startpoint),start:end]=np.maximum(wall[l+((n1-1)*vspacing+startpoint),start:end],testcrop[differencestart:jointcropwidth-differenceend])
            
    mask = (wall > 0.05).astype(int)
    
  #  maskim = Image.fromarray(mask)
  #  maskim.save(saveloc)
    
    return wall,mask

    