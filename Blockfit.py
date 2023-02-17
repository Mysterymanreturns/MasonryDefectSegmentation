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
       # print('x,a')
       # print(hjointloc[X])
       # print(a)
        
        if hjointloc[X] == True and a == False:
            endx[X] = 1
        #    print("end")
        if hjointloc[X] == False and a == True:
            startx[X] = 1
       #     print("start")
        a = hjointloc[X]
    endx[0]=0
    endx[-1]=1

    startindex = startx.nonzero()
    endindex = endx.nonzero()
    if startindex[0] > endindex[0]:
        startindex = torch.cat((torch.zeros(1).unsqueeze(0), startindex))
    if startindex[-1] > endindex[-1]:
        endindex = torch.cat((endindex, (startx.size()[0]-1)*torch.ones(1).unsqueeze(0)))
    x2 = hjoints
    widthadd = 1
    widthcheck = widthadd*2 +1
    for start, end in zip(startindex,endindex):
        vjoints = 0*torch.sum(x[0][int(start.item()):int(end.item()),:],dim =0)
        for n in range(0, widthcheck):
            vjoints[(widthadd):(-widthadd)] += torch.sum(x[0][int(start.item()):int(end.item()),(0+n):x[0].shape[1]-widthcheck+n+1],dim =0)/(widthcheck*(end.item()-start.item()))

        THRESHOLD = min(torch.mean(vjoints)+torch.std(vjoints),0.99)
        vjoints = vjoints> 0.5 #THRESHOLD
        vjointloc = vjoints
        vjoints = torch.unsqueeze(vjoints,0)
        vjoints = vjoints*torch.ones(x[0][int(start.item()):int(end.item()),:].shape)
        x2[int(start.item()):int(end.item()),:] = vjoints + hjoints[int(start.item()):int(end.item()),:]#

    fit = np.array(x2)



    return fit