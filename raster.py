def raster(fileloc,saveloc,mask = 0,res = 1, dim = None):
    
    
    from PIL import Image, ImageOps

    import os
    import cv2
    import laspy
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    if mask == 0 and "tiff" not in fileloc:
        las = laspy.read(fileloc)
        print(np.unique(las.classification)) 
        for dimension in las.point_format.dimensions:
            print(dimension.name)
        points = np.vstack((las.X, las.Y, las.Z)).transpose()    
        nop = len(points[:,0])
        print('number of points = {}'.format(nop))
        xmin = np.argmin(points[:,0])
        xminv= points[xmin,0]
        xmax = np.argmax(points[:,0])
        xmaxv= points[xmax,0]
        ymin = np.argmin(points[:,1])
        yminv= points[ymin,1]
        ymax = np.argmax(points[:,1])
        ymaxv= points[ymax,1]
        ywidth =ymaxv-yminv
        xwidth =xmaxv-xminv
        ratio = xwidth/ywidth
        print(ywidth,xwidth,ratio)
        nx = (ratio*nop)**(1/2)
        nx = nx*res

        nx = int(nx)
        xbinsize = (xmaxv-xminv)/nx
        ybinsize = xbinsize
        ny = (ymaxv-yminv)/ybinsize
        ny = int(ny)+1
        nx = nx+1

        image = np.zeros((nx,ny))
        npix = image.shape[0]*image.shape[1]
        print('number of pixels = {}'.format(npix))
        print('width in pixels = {}'.format(nx))
        print('height in pixels = {}'.format(ny))
        xp = np.floor((points[:,0]-xminv)/xbinsize).astype(int)
        yp = np.floor((points[:,1]-yminv)/ybinsize).astype(int)
        zp = points[:,2]
        image = np.zeros((nx,ny))
        imagenum = np.zeros((nx,ny))
        for i in range(len(xp)):
            imagenum[xp[i],yp[i]]+=1
        a = 0
        for i in range(len(xp)):
            a = imagenum[xp[i],yp[i]]
         #   if a != 0:
            image[xp[i],yp[i]]+=zp[i]/a
        image = image - image.min()
        imagenumb = imagenum.astype(bool)
        imagenumb = 0 -imagenumb
        plt.imshow(imagenumb)
        plt.imshow(image)
       # outputnum2 = Image.fromarray(imagenumb)
       # outputnum2 = outputnum2.convert("L")
        #imagenumb2 = np.array(outputnum2)
        #output = Image.fromarray(image)
        #image2 = np.array(output)
        #infilled = cv2.inpaint(image2, imagenum.astype('uint8'), 3, cv2.INPAINT_NS)
        for a in range(len(imagenum[0,:])):
            for b in range(len(imagenum[:,0])):
                if imagenum[b,a] == 0:
                   c = 0
                   d = 0
                   E = [-1, 0, 1]
                   F = [-1, 0, 1]
                   for e in E:
                        for f in F:
                           try:
                                if imagenum[b+e,a+f] != 0:
                                    d+=image[b+e,a+f]
                                    c+=1
                           except:
                                pass
                   if d !=0:

                       image[b,a]=d/c

     #   for a in range(len(imagenum[0,:])-1,0,-1):
     #       for b in range(len(imagenum[:,0])-1,0,-1):
     #           if imagenum[b,a] == 0:#
    #tese
    #                try:
    #                    image[b,a]=image[b+1,a]
    #                except:
    #                    b   


        outputf = Image.fromarray(image)
        print('mean pixel value = {}'.format(image.mean()))
        outputf.save(saveloc+"raster.tiff")
    #np.save(saveloc+"raster.tiff",image)
    elif mask == 1:
        image = Image.open(saveloc+"raster.tiff")
        nx,ny = image.size
        outputf = Image.open(fileloc)
        outputf = outputf.resize((nx,ny))
        outputf.save(saveloc+"mask.tiff")
    elif mask == 0 and "tiff" in fileloc:
        outputf = Image.open(fileloc)
        outputf.save(saveloc+"raster.tiff")        
    noim = 0
    if dim != None:
            if mask == 0:
                newpath = r'croppedimages/'
            else:
                newpath = r'croppedmasks/'
            if not os.path.exists(saveloc+newpath):
                os.makedirs(saveloc+newpath)
            pic = outputf
            #pic = ImageOps.grayscale(pic)
            w = pic.size[0]
            nx = int(w/dim)
            h = pic.size[1]
            ny = int(h/dim)
            i = 0
            print("cropping masks")
            for x in range(0, nx):
                for y in range(0, ny):
                  i = i+1
                  box = (x*dim, y*dim, dim*(1+x) , dim*(1+y))
                  region = pic.crop(box)
                  #region.save("test2/croppedimagestargetfullres2/"+ str(i) +".png")
                  region.save(saveloc+newpath+ str(i) +".tiff")
                  noim +=1
                  


    return noim