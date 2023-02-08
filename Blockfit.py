def blockfit(data):


    import numpy as np
    import torch
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    trans = A.Compose([
            ToTensorV2()])
    
    data = data/data.max()
    transformed = trans(image=data, mask=data)
    x = transformed['image']
    x = x*data.max()


    hjoints = torch.sum(x[0],dim =1)/x[0].shape[1]

    THRESHOLD = min(torch.mean(hjoints)+torch.std(hjoints),0.99)
    hjoints = hjoints>THRESHOLD
    hjointloc = hjoints
    hjoints = torch.unsqueeze(hjoints,1)
    hjoints = hjoints*torch.ones(x[0].shape)

    x1 = x[0]+hjoints

    a= False
    startx = hjointloc*0
    endx = hjointloc*0
    for X in range(0,len(hjointloc)):
        print('x,a')
        print(hjointloc[X])
        print(a)
        
        if hjointloc[X] == True and a == False:
            endx[X] = 1
            print("end")
        if hjointloc[X] == False and a == True:
            startx[X] = 1
            print("start")
        a = hjointloc[X]
    endx[0]=0
    endx[-1]=1

    startindex = startx.nonzero()
    endindex = endx.nonzero()

    x2 = hjoints
    widthadd = 1
    widthcheck = widthadd*2 +1
    for start, end in zip(startindex,endindex):
        vjoints = 0*torch.sum(x[0][start:end,:],dim =0)
        for n in range(0, widthcheck):
            vjoints[(widthadd):(-widthadd)] += torch.sum(x[0][start:end,(0+n):x[0].shape[1]-widthcheck+n+1],dim =0)/(widthcheck*(end-start))

        THRESHOLD = min(torch.mean(vjoints)+torch.std(vjoints),0.99)
        vjoints = vjoints>THRESHOLD
        vjointloc = vjoints
        vjoints = torch.unsqueeze(vjoints,0)
        vjoints = vjoints*torch.ones(x[0][start:end,:].shape)
        x2[start:end,:] = vjoints + hjoints[start:end,:]

    fit = np.array(x2)



    return fit