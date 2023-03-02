def postprocess(blocksave,spallsave,imsave):

    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import cv2 as cv
    import scipy.linalg

    pic = Image.open(blocksave)
    picnp = np.array(pic)
    picspall = Image.open(spallsave)
    picnpspall = np.array(picspall)
    spallmask = picnpspall<150
    picoriginal = Image.open(imsave)
    picorignp = np.array(picoriginal)
    labels, markedim = cv.connectedComponents(255-picnp)
    labeledimage = picorignp*0
    print(labels)
    for blockno in range(1,labels):
       # try:
            blockextract = (markedim == blockno).astype(np.uint8)
            block = blockextract*picorignp
            labelst = np.nonzero(block)
            blockmask = block*spallmask
            blockcrop = blockmask[labelst[0].min():labelst[0].max(),labelst[1].min():labelst[1].max()]


            data = blockcrop.flatten()#np.random.multivariate_normal(mean, cov, 50)
            unspalledindex = np.nonzero(data)
            # regular grid covering the domain of the data
            X,Y = np.meshgrid(np.arange(0,blockcrop.shape[1],1), np.arange(0,blockcrop.shape[0],1))
            XX = X.flatten()
            YY = Y.flatten()
            data = data[unspalledindex]
            XX = XX[unspalledindex]
            YY = YY[unspalledindex]
            order = 1    # 1: linear, 2: quadratic
            if order == 1:
                # best-fit linear plane
                A = np.c_[XX, YY, np.ones(data.shape[0])]
                C,_,_,_ = scipy.linalg.lstsq(A, data)    # coefficients
                
                # evaluate it on grid
                Z = C[0]*X + C[1]*Y + C[2]

            blockoffset = block[labelst[0].min():labelst[0].max(),labelst[1].min():labelst[1].max()]-Z
            labeledimage[labelst[0].min():labelst[0].max(),labelst[1].min():labelst[1].max()] = blockoffset
       # except:
            continue

    #need to add classification

    return labeledimage