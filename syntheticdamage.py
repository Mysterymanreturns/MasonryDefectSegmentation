def syntheticspall(wall,mask,damagelevel):  
    
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    import numpy as np
    import albumentations as A
    from skimage.io import imread
    from scipy.special import binom
    from skimage import feature
    from skimage import segmentation
    from skimage import morphology
    from scipy import ndimage
    import cv2 as cv
    from scipy import ndimage
    from skimage import segmentation
    import random

    bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

    def bezier(points, num=200):
        N = len(points)
        t = np.linspace(0, 1, num=num)
        curve = np.zeros((num, 2))
        for i in range(N):
            curve += np.outer(bernstein(N - 1, i, t), points[i])
        return curve

    class Segment():
        def __init__(self, p1, p2, angle1, angle2, **kw):
            self.p1 = p1; self.p2 = p2
            self.angle1 = angle1; self.angle2 = angle2
            self.numpoints = kw.get("numpoints", 100)
            r = kw.get("r", 0.3)
            d = np.sqrt(np.sum((self.p2-self.p1)**2))
            self.r = r*d
            self.p = np.zeros((4,2))
            self.p[0,:] = self.p1[:]
            self.p[3,:] = self.p2[:]
            self.calc_intermediate_points(self.r)

        def calc_intermediate_points(self,r):
            self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                        self.r*np.sin(self.angle1)])
            self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                        self.r*np.sin(self.angle2+np.pi)])
            self.curve = bezier(self.p,self.numpoints)


    def get_curve(points, **kw):
        segments = []
        for i in range(len(points)-1):
            seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
            segments.append(seg)
        curve = np.concatenate([s.curve for s in segments])
        return segments, curve

    def ccw_sort(p):
        d = p-np.mean(p,axis=0)
        s = np.arctan2(d[:,0], d[:,1])
        return p[np.argsort(s),:]

    def get_bezier_curve(a, rad=0.2, edgy=0):
        """ given an array of points *a*, create a curve through
        those points. 
        *rad* is a number between 0 and 1 to steer the distance of
              control points.
        *edgy* is a parameter which controls how "edgy" the curve is,
               edgy=0 is smoothest."""
        p = np.arctan(edgy)/np.pi+.5
        a = ccw_sort(a)
        a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
        d = np.diff(a, axis=0)
        ang = np.arctan2(d[:,1],d[:,0])
        f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
        ang = f(ang)
        ang1 = ang
        ang2 = np.roll(ang,1)
        ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
        ang = np.append(ang, [ang[0]])
        a = np.append(a, np.atleast_2d(ang).T, axis=1)
        s, c = get_curve(a, r=rad, method="var")
        x,y = c.T
        return x,y, a


    def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
        """ create n random points in the unit square, which are *mindst*
        apart, then scale them."""
        mindst = mindst or .7/n
        a = np.random.rand(n,2)
        d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
        if np.all(d >= mindst) or rec>=200:
            return a*scale
        else:
            return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)


    # characterise wall
    spalledwall = wall
    wall2 = wall
    #dtransformwall = ndimage.distance_transform_edt(wall2)
    #labeledwall = segmentation.watershed(dtransformwall, watershed_line=True)
 #   noblocks = labeledwall.max()
    out=cv.connectedComponents(1-mask.astype("uint8"))
    labeledwall = out[1]
    noblocks = out[0]
    
    
    #random transforms in each block
    for n1 in range(noblocks-1):
       # n1=n1
        #print(n1)
        block = (labeledwall==n1)
        block.astype("int")
        
        blockindex = np.transpose(np.nonzero(block))
        blockxstart=blockindex[:,0].min()
        blockxend=blockindex[:,0].max()
        blockystart=blockindex[:,1].min()
        blockyend=blockindex[:,1].max()
        blocklength = blockxend-blockxstart
        blockheight = blockyend-blockystart 
        thisblock = block[blockxstart:blockxend,blockystart:blockyend]
        
        if  (random.random() < damagelevel)==1:
            for n2 in range(random.randint(0, int(25*damagelevel))):
                
                #random depression spalling
        
                rad = random.random()
                edgy = random.random()*10
                npoints = random.randint(3,15)
                for c in np.array([[0,0]]):

                    a = get_random_points(npoints, scale=1) + c
                    x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)


                x=(x*100).astype(int)
                y=(y*100).astype(int)
               # plt.plot(x,y)

                spall = np.zeros((105,105))

                for n in range(len(x)):
                  a = min(x[n]+3,102)
                  b = min(y[n]+3,102)
                  spall[a,b]=1

                spall=spall.astype(int)
                cv.floodFill(spall, None, (0,0),1)
                spall=1-spall
                spallD = ndimage.distance_transform_edt(spall)
                spall = spallD*100/(max(1,spallD.max())*random.randint(50, 200))
                
                spall = spall**(0.1*random.randint(1, 50))
                
                
                #if  (random.random() < damagelevel)==1:
                
                #extra flat spalling
                for n3 in range(random.randint(0, 10)):
                    rad = random.random()
                    edgy = random.random()*10
                    npoints = random.randint(3,15)

                    for c2 in np.array([[0,0]]):

                        a2 = get_random_points(npoints, scale=1) + c2
                        x2,y2, _ = get_bezier_curve(a2,rad=rad, edgy=edgy)


                    x2=(x2*100).astype(int)
                    y2=(y2*100).astype(int)
                   # plt.plot(x,y)

                    spallflat = np.zeros((105,105))

                    for n4 in range(len(x2)):
                      a2 = min(x2[n4]+3,102)
                      b2 = min(y2[n4]+3,102)
                      spallflat[a2,b2]=1

                    spallflat=spallflat.astype(int)
                    cv.floodFill(spallflat, None, (0,0),1)
                    #spallflat = np.array(spallflat)
                    spallflat = 1-spallflat
                    spallflat = spallflat.astype("uint8")
                    spallflatsize =  int(100/random.randint(2, 20))
                    
                    spallflatC = cv.resize(spallflat,(spallflatsize,spallflatsize))
                    spallflat = spallflatC
                    
                    flatlocx = random.randint(0, 100-spallflatsize)
                    flatlocy = random.randint(0, 100-spallflatsize)
                    spallflatintensity =spall[flatlocx+int(spallflatsize/2),flatlocy+int(spallflatsize/2)]
                    spallflat = spallflat*spallflatintensity

                    
                    spall[flatlocx:flatlocx+spallflatsize,flatlocy:flatlocy+spallflatsize] = np.where(spallflat>0,spallflat,spall[flatlocx:flatlocx+spallflatsize,flatlocy:flatlocy+spallflatsize])
                    

                
                spalllength = int(blocklength*100/random.randint(25, 600))
                spallheight = int(blockheight*100/random.randint(25, 600))
                spallC = cv.resize(spall,(spallheight,spalllength))
                spall = spallC
                spalllength = spall.shape[0]
                spallheight = spall.shape[1]
                locx = blockxstart-spalllength+random.randint(0, blocklength+2*spalllength)
                locy =blockystart-spallheight+random.randint(0, blockheight+2*spallheight)
                overlapxstart=max(0,blockxstart-locx)
                overlapxend=max(0,locx+spalllength-blockxend)
                overlapystart=max(0,blockystart-locy)
                overlapyend=max(0,locy+spallheight-blockyend)
                
                defect = spall[overlapxstart:max(0,spalllength-overlapxend),overlapystart:max(0,spallheight-overlapyend)]
                #defect = defect.astype("float32")
               # defect = cv.GaussianBlur(defect,(5,5),0)
                defectlength = defect.shape[0]
                defectheight = defect.shape[1]
                spalledwall[locx+overlapxstart:locx+overlapxstart+defectlength,locy+overlapystart:locy+defectheight+overlapystart] = np.maximum(defect,spalledwall[locx+overlapxstart:locx+overlapxstart+defectlength,locy+overlapystart:locy+defectheight+overlapystart])
                
    print('damage added')
   #spallingmask = spalledwall>0
    return spalledwall#,spallingmask



def syntheticflor(wall,mask,damagelevel):  
    
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    import numpy as np
    import albumentations as A
    from skimage.io import imread
    from scipy.special import binom
    from skimage import feature
    from skimage import segmentation
    from skimage import morphology
    from scipy import ndimage
    import cv2 as cv
    from scipy import ndimage
    from skimage import segmentation
    import random

    bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

    def bezier(points, num=200):
        N = len(points)
        t = np.linspace(0, 1, num=num)
        curve = np.zeros((num, 2))
        for i in range(N):
            curve += np.outer(bernstein(N - 1, i, t), points[i])
        return curve

    class Segment():
        def __init__(self, p1, p2, angle1, angle2, **kw):
            self.p1 = p1; self.p2 = p2
            self.angle1 = angle1; self.angle2 = angle2
            self.numpoints = kw.get("numpoints", 100)
            r = kw.get("r", 0.3)
            d = np.sqrt(np.sum((self.p2-self.p1)**2))
            self.r = r*d
            self.p = np.zeros((4,2))
            self.p[0,:] = self.p1[:]
            self.p[3,:] = self.p2[:]
            self.calc_intermediate_points(self.r)

        def calc_intermediate_points(self,r):
            self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                        self.r*np.sin(self.angle1)])
            self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                        self.r*np.sin(self.angle2+np.pi)])
            self.curve = bezier(self.p,self.numpoints)


    def get_curve(points, **kw):
        segments = []
        for i in range(len(points)-1):
            seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
            segments.append(seg)
        curve = np.concatenate([s.curve for s in segments])
        return segments, curve

    def ccw_sort(p):
        d = p-np.mean(p,axis=0)
        s = np.arctan2(d[:,0], d[:,1])
        return p[np.argsort(s),:]

    def get_bezier_curve(a, rad=0.2, edgy=0):
        """ given an array of points *a*, create a curve through
        those points. 
        *rad* is a number between 0 and 1 to steer the distance of
              control points.
        *edgy* is a parameter which controls how "edgy" the curve is,
               edgy=0 is smoothest."""
        p = np.arctan(edgy)/np.pi+.5
        a = ccw_sort(a)
        a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
        d = np.diff(a, axis=0)
        ang = np.arctan2(d[:,1],d[:,0])
        f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
        ang = f(ang)
        ang1 = ang
        ang2 = np.roll(ang,1)
        ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
        ang = np.append(ang, [ang[0]])
        a = np.append(a, np.atleast_2d(ang).T, axis=1)
        s, c = get_curve(a, r=rad, method="var")
        x,y = c.T
        return x,y, a


    def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
        """ create n random points in the unit square, which are *mindst*
        apart, then scale them."""
        mindst = mindst or .7/n
        a = np.random.rand(n,2)
        d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
        if np.all(d >= mindst) or rec>=200:
            return a*scale
        else:
            return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)


    # characterise wall
    spalledwall = wall
    wall2 = wall
    
    
    for n2 in range(random.randint(0, int(5*damagelevel))):
                
                #random depression spalling
        
                rad = random.random()
                edgy = random.random()*10
                npoints = random.randint(3,15)
                for c in np.array([[0,0]]):

                    a = get_random_points(npoints, scale=1) + c
                    x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)


                x=(x*100).astype(int)
                y=(y*100).astype(int)
               # plt.plot(x,y)

                spall = np.zeros((105,105))

                for n in range(len(x)):
                  a = min(x[n]+3,102)
                  b = min(y[n]+3,102)
                  spall[a,b]=1

                spall=spall.astype(int)
                cv.floodFill(spall, None, (0,0),1)
                spall=1-spall
                spallD = ndimage.distance_transform_edt(spall)
                spall = spallD*100/(max(1,spallD.max())*random.randint(50, 200))

                
                spalllength = int(wall.shape[0]*10/random.randint(10, 200))
                spallheight = int(wall.shape[1]*10/random.randint(10, 200))
                spallC = cv.resize(spall,(spallheight,spalllength))
                spall = spallC
                
                efflor = spall*0
                efflor = addperlin3(efflor,10/random.randint(5, 100))
                spall = spall*efflor
                                
                spalllength = spall.shape[0]
                spallheight = spall.shape[1]
                locx = -spalllength+random.randint(0, blocklength+2*spalllength)
                locy = -spallheight+random.randint(0, blockheight+2*spallheight)
                overlapxstart=max(0,-locx)
                overlapxend=max(0,locx+spalllength-wall.shape[0])
                overlapystart=max(0,-locy)
                overlapyend=max(0,locy+spallheight-wall.shape[1])
                
                defect = spall[overlapxstart:max(0,spalllength-overlapxend),overlapystart:max(0,spallheight-overlapyend)]
                #defect = defect.astype("float32")
               # defect = cv.GaussianBlur(defect,(5,5),0)
                defectlength = defect.shape[0]
                defectheight = defect.shape[1]
                #spalledwall[locx+overlapxstart:locx+overlapxstart+defectlength,locy+overlapystart:locy+defectheight+overlapystart] = np.maximum(defect,spalledwall[locx+overlapxstart:locx+overlapxstart+defectlength,locy+overlapystart:locy+defectheight+overlapystart])
                spalledwall[locx+overlapxstart:locx+overlapxstart+defectlength,locy+overlapystart:locy+defectheight+overlapystart] = spalledwall[locx+overlapxstart:locx+overlapxstart+defectlength,locy+overlapystart:locy+defectheight+overlapystart]-defect
    
    
    
    
    
    
    
    #dtransformwall = ndimage.distance_transform_edt(wall2)
    #labeledwall = segmentation.watershed(dtransformwall, watershed_line=True)
 #   noblocks = labeledwall.max()
   # out=cv.connectedComponents(1-mask.astype("uint8"))
   # labeledwall = out[1]
   # noblocks = out[0]
    
    
    #random transforms in each block
#     for n1 in range(noblocks-1):
#        # n1=n1
#         #print(n1)
#         block = (labeledwall==n1)
#         block.astype("int")
        
#         blockindex = np.transpose(np.nonzero(block))
#         blockxstart=blockindex[:,0].min()
#         blockxend=blockindex[:,0].max()
#         blockystart=blockindex[:,1].min()
#         blockyend=blockindex[:,1].max()
#         blocklength = blockxend-blockxstart
#         blockheight = blockyend-blockystart 
#         thisblock = block[blockxstart:blockxend,blockystart:blockyend]
        
#         if  (random.random() < damagelevel)==1:
#             for n2 in range(random.randint(0, int(25*damagelevel))):
                
#                 #random depression spalling
        
#                 rad = random.random()
#                 edgy = random.random()*10
#                 npoints = random.randint(3,15)
#                 for c in np.array([[0,0]]):

#                     a = get_random_points(npoints, scale=1) + c
#                     x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)


#                 x=(x*100).astype(int)
#                 y=(y*100).astype(int)
#                # plt.plot(x,y)

#                 spall = np.zeros((105,105))

#                 for n in range(len(x)):
#                   a = min(x[n]+3,102)
#                   b = min(y[n]+3,102)
#                   spall[a,b]=1

#                 spall=spall.astype(int)
#                 cv.floodFill(spall, None, (0,0),1)
#                 spall=1-spall
#                 spallD = ndimage.distance_transform_edt(spall)
#                 spall = spallD*100/(max(1,spallD.max())*random.randint(50, 200))
                
                #if  (random.random() < damagelevel)==1:
                
                #extra flat spalling
#                 for n3 in range(random.randint(0, 10)):
#                     rad = random.random()
#                     edgy = random.random()*10
#                     npoints = random.randint(3,15)

#                     for c2 in np.array([[0,0]]):

#                         a2 = get_random_points(npoints, scale=1) + c2
#                         x2,y2, _ = get_bezier_curve(a2,rad=rad, edgy=edgy)


#                     x2=(x2*100).astype(int)
#                     y2=(y2*100).astype(int)
#                    # plt.plot(x,y)

#                     spallflat = np.zeros((105,105))

#                     for n4 in range(len(x2)):
#                       a2 = min(x2[n4]+3,102)
#                       b2 = min(y2[n4]+3,102)
#                       spallflat[a2,b2]=1

#                     spallflat=spallflat.astype(int)
#                     cv.floodFill(spallflat, None, (0,0),1)
#                     #spallflat = np.array(spallflat)
#                     spallflat = 1-spallflat
#                     spallflat = spallflat.astype("uint8")
#                     spallflatsize =  int(100/random.randint(2, 20))
                    
#                     spallflatC = cv.resize(spallflat,(spallflatsize,spallflatsize))
#                     spallflat = spallflatC
                    
#                     flatlocx = random.randint(0, 100-spallflatsize)
#                     flatlocy = random.randint(0, 100-spallflatsize)
#                     spallflatintensity =spall[flatlocx+int(spallflatsize/2),flatlocy+int(spallflatsize/2)]
#                     spallflat = spallflat*spallflatintensity

                    
#                     spall[flatlocx:flatlocx+spallflatsize,flatlocy:flatlocy+spallflatsize] = np.where(spallflat>0,spallflat,spall[flatlocx:flatlocx+spallflatsize,flatlocy:flatlocy+spallflatsize])
                    

                
#                 spalllength = int(blocklength*100/random.randint(25, 600))
#                 spallheight = int(blockheight*100/random.randint(25, 600))
#                 spallC = cv.resize(spall,(spallheight,spalllength))
#                 spall = spallC
#                 spalllength = spall.shape[0]
#                 spallheight = spall.shape[1]
#                 locx = blockxstart-spalllength+random.randint(0, blocklength+2*spalllength)
#                 locy =blockystart-spallheight+random.randint(0, blockheight+2*spallheight)
#                 overlapxstart=max(0,blockxstart-locx)
#                 overlapxend=max(0,locx+spalllength-blockxend)
#                 overlapystart=max(0,blockystart-locy)
#                 overlapyend=max(0,locy+spallheight-blockyend)
                
#                 defect = spall[overlapxstart:max(0,spalllength-overlapxend),overlapystart:max(0,spallheight-overlapyend)]
#                 #defect = defect.astype("float32")
#                # defect = cv.GaussianBlur(defect,(5,5),0)
#                 defectlength = defect.shape[0]
#                 defectheight = defect.shape[1]
#                 spalledwall[locx+overlapxstart:locx+overlapxstart+defectlength,locy+overlapystart:locy+defectheight+overlapystart] = np.maximum(defect,spalledwall[locx+overlapxstart:locx+overlapxstart+defectlength,locy+overlapystart:locy+defectheight+overlapystart])
                
    print('floresced')
   #spallingmask = spalledwall>0
    return spalledwall#,spallingmask
