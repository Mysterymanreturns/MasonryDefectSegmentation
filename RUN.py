
# INPUTS
#inno = number of input channels
#savestate = trained model to load in. format : 'test+"/results/"+savestate'
#type1 = type of data used eg. depth, normal, RGB
#network = type of network eg. Unet
#test = path to data folder
#encoder = name of encoder eg. mobilenet_v2
#dim = dimension of image crops used in network. Images will be square. eg. dim = 512 will mean 512x512 patches. (note this code will autocrop input images into patches of the correct size)
# testimage = name of testimage. Note, should be in root of data folder defined by 'test'
# testmask = name of mask for testimage. Note, should be in root of data folder defined by 'test'

# INPUT FILES
# pytorch model at location: test+"/results/"+savestate
# test image at location: test+testimage

#OUTPUTS
# iou_score = Intersection over union score for test image
# precision = precisions score for test image
# recall = recall score for test image

# OUTPUT FILES
# output of network on testimage after sigmoid function at location: test+'/results/picout{}{}{}{}.tiff'.format(type1,network,encoder,savetag)


def UNETrun(inno, savestate, type1, network, test, encoder, dim, testimage, testmask, savetag):

    import numpy as np
    from Blockfit import blockfit
    import torch
    import torchvision
    from torch import nn, optim
    from torchvision import datasets
    import matplotlib.pyplot as plt
    import math
    from PIL import Image, ImageOps    
    from torch.utils import data
    import albumentations as A
    import torchvision.utils as vutils
    from torch.utils.data import Dataset as BaseDataset
    import torchvision.transforms as transforms
    from sklearn.model_selection import train_test_split
    import pathlib
    from torch.autograd import Variable
    import torch.nn.functional as F
    from albumentations.pytorch import ToTensorV2    
    import segmentation_models_pytorch as smp

    Net = getattr(smp, network)(
        encoder_name= encoder,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
 #   encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=inno,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
        ) 

    d = torch.load(test+"/results/"+savestate)
    Net.load_state_dict(d["state_dict"])

   # Net.eval()
    Net.cuda()
    from PIL import Image
   # if inno == 1:
   #     pic = Image.open(test).convert('L')
   # if inno == 3:
   #     pic = Image.open(test).convert('RGB')
    
    pic = Image.open(test+testimage)
    
    
    def normaliseimg(normalisationparam, picnormnp):
            
        picnormnp = ((picnormnp-picnormnp.mean())/(normalisationparam*np.std(picnormnp)))+0.5
        picnormnp[picnormnp<0] = 0
        picnormnp[picnormnp>1] = 1
        
        return picnormnp
    normalisationparam = 4    
    picnormnp = np.array(pic)
    picnormnp = normaliseimg(normalisationparam,picnormnp)
    pic = Image.fromarray(picnormnp)
    
    
    
    pic1 = pic.copy()
    pic2 = pic.copy()
    pic3 = pic.copy()
    pic4 = pic.copy()
   # dim = 512
    w = pic.size[0]
    nx = int(w/dim)
    h = pic.size[1]
    ny = int(h/dim)
    i = 0
    
    picpastenp1 = np.zeros((w,h))
    picpaste1 = Image.new(mode = 'F',size=(w,h)) 
    picpaste2 = Image.new(mode = 'F',size=(w,h))
    picpaste3 = Image.new(mode = 'F',size=(w,h))
    co_ords = np.zeros((2, nx*ny))

    transformpil = transforms.ToPILImage()
    size = 256

    transform1 = A.Compose([
           # A.PadIfNeeded(min_height=512, min_width=512, p=1),
           # A.Resize(size,size),
            ToTensorV2(),
            ])    
    offset = 0
    if nx == (w-offset)/dim:
        nxend = nx
    else:
        nxend = nx+1

    if ny == (h-offset)/dim:
        nyend = ny
    else:
        nyend = ny+1
        
    for x in range(0, nxend):
        for y in range(0,nyend):
          left = int(x*dim)
          bottom = int(y*dim)
          right =int(dim*(1+x))
          top =int(dim*(1+y))
          box = (left,bottom,right,top)
          region = pic.crop(box)


          if left< 0:

            overlapleft = 0 - left
            end = max(dim,(2*overlapleft))
            copy = region.crop((overlapleft,0,end, dim))
            width = copy.size[0]
            mirrored = copy.transpose(0)
            noverlap = int(overlapleft/width)
            for a in range(1, noverlap+1):
                region.paste(mirrored,(end-width*(a+1),0,end-width*a,dim) )


          if right> w:


            overlapright = right - w
            start = max(0,dim-(2*overlapright))
            copy = region.crop((start,0,(dim-overlapright), dim))
            width = copy.size[0]
            mirrored = copy.transpose(0)
            noverlap = int(overlapright/width)
            for a in range(1, noverlap+1):
                region.paste(mirrored,(start+width*a,0,start+width*(1+a),dim) )



          if bottom< 0:

            overlapbottom = 0 - bottom
            end = max(dim,(2*overlapbottom))
            copy = region.crop((0,overlapbottom,dim,end))
            height = copy.size[1]
            mirrored = copy.transpose(1)
            noverlap = int(overlapbottom/height)
            for a in range(1, noverlap+1):
                region.paste(mirrored,(0,end-height*(a+1),dim,end-height*a) )

          if top> h:

            
            overlaptop = top - h
            start = max(0,dim-(2*overlaptop))
            copy = region.crop((0,start,dim,(dim-overlaptop)))
            height = copy.size[1]
            mirrored = copy.transpose(1)
            noverlap = int(overlaptop/height)
            for a in range(1, noverlap+1):
                region.paste(mirrored,(0,start+height*a,dim,start+height*(1+a)) )



       
          with torch.no_grad():
         
                  imagepre = np.array(region)

                  image = imagepre#.astype(np.uint8)
                  image = normaliseimg(normalisationparam,image)
                  transformed = transform1(image=image)
                  image = transformed["image"]

                  img = image.float().cuda()
                 
                  img = torch.unsqueeze(img,0)
                  pred = Net(img)

                

                  pred = pred.type(torch.FloatTensor)
                  

                
                  out = pred.cpu().detach().numpy()
                 # print(out[0][0].min())
                  output = Image.fromarray(out[0][0])

                
                  outputf = output.resize((dim,dim))  
                  
               #   outputfnp = np.array(outputfnp)
                    
                  picpaste1.paste(outputf, box)  
                  #pic1.paste(outputf, box)
                #  t = np.array(pic1)
                 # picpastenp1[]
                 # print(outputf.mode)
                 # print(t.min())
                    
                    
                  #img = img.cpu().detach().numpy()  
                  #plt.imshow(img[0][0], cmap='gray') 
                #  plt.imshow(out[0][0], cmap='gray') 

   # plt.imshow(pic1, cmap='gray') 
   # plt.imshow(picpaste1, cmap='gray') 
   # plt.imshow(pic1, cmap='gray') 
    offset = -dim/2
    

    nx2 = int((w-offset)/dim)
    ny2 = int((h-offset)/dim)
    
    if nx2 == (w-offset)/dim:
        nxend2 = nx2
    else:
        nxend2 = nx2+1

    if ny2 == (h-offset)/dim:
        nyend2 = ny2
    else:
        nyend2 = ny2+1

    for x in range(0, nxend2):
        for y in range(0,nyend2):

          left = int(x*dim+offset)
          bottom = int(y*dim+offset)
          right =int(dim*(1+x)+offset)
          top =int(dim*(1+y)+offset)
          box = (left,bottom,right,top)
          region = pic.crop(box)


          if left< 0:


            overlapleft = 0 - left
            end = max(dim,(2*overlapleft))
            copy = region.crop((overlapleft,0,end, dim))
            width = copy.size[0]
            mirrored = copy.transpose(0)
            noverlap = int(overlapleft/width)
            for a in range(1, noverlap+1):
                region.paste(mirrored,(end-width*(a+1),0,end-width*a,dim) )


          if right> w:


            overlapright = right - w
            start = max(0,dim-(2*overlapright))
            copy = region.crop((start,0,(dim-overlapright), dim))
            width = copy.size[0]
            mirrored = copy.transpose(0)
            noverlap = int(overlapright/width)
            for a in range(1, noverlap+1):
                region.paste(mirrored,(start+width*a,0,start+width*(1+a),dim) )



          if bottom< 0:

            overlapbottom = 0 - bottom
            end = max(dim,(2*overlapbottom))
            copy = region.crop((0,overlapbottom,dim,end))
            height = copy.size[1]
            mirrored = copy.transpose(1)
            noverlap = int(overlapbottom/height)
            for a in range(1, noverlap+1):
                region.paste(mirrored,(0,end-height*(a+1),dim,end-height*a) )

          if top> h:

            overlaptop = top - h
            start = max(0,dim-(2*overlaptop))
            copy = region.crop((0,start,dim,(dim-overlaptop)))
            height = copy.size[1]
            mirrored = copy.transpose(1)
            noverlap = int(overlaptop/height)
            for a in range(1, noverlap+1):
                region.paste(mirrored,(0,start+height*a,dim,start+height*(1+a)) )        



       
          with torch.no_grad():
         
                  imagepre = np.array(region)

                  image = imagepre#.astype(np.uint8)
                  image = normaliseimg(normalisationparam,image)
                  transformed = transform1(image=image)
                  image = transformed["image"]

                  img = image.float().cuda()
                 
                  img = torch.unsqueeze(img,0)
                  pred = Net(img)

                

                  pred = pred.type(torch.FloatTensor)
                  

                
                  out = pred.cpu().detach().numpy()
                  output = Image.fromarray(out[0][0])

                
                  outputf = output.resize((dim,dim)) 
                  picpaste2.paste(outputf, box) 
                  #pic2.paste(outputf, box)

    offset = int(-dim/5)


    size = 256
    nx3 = int((w-offset)/dim)
    ny3 = int((h-offset)/dim)
    
    if nx3 == (w-offset)/dim:
        nxend3 = nx3
    else:
        nxend3 = nx3+1

    if ny3 == (h-offset)/dim:
        nyend3 = ny3
    else:
        nyend3 = ny3+1

    
    for x in range(0, nxend3):
        for y in range(0,nyend3):

          left = int(x*dim+offset)
          bottom = int(y*dim+offset)
          right =int(dim*(1+x)+offset)
          top =int(dim*(1+y)+offset)
          box = (left,bottom,right,top)
          region = pic.crop(box)


          if left< 0:

            overlapleft = 0 - left
            end = min(dim,(2*overlapleft))
            copy = region.crop((overlapleft,0,end, dim))
            width = copy.size[0]
            mirrored = copy.transpose(0)
            noverlap = int(overlapleft/width)
            for a in range(1, noverlap+1):
                region.paste(mirrored,(end-width*(a+1),0,end-width*a,dim) )


          if right> w:


            overlapright = right - w
            start = max(0,dim-(2*overlapright))
            copy = region.crop((start,0,(dim-overlapright), dim))
            width = copy.size[0]
            mirrored = copy.transpose(0)
            noverlap = int(overlapright/width)
            for a in range(1, noverlap+1):
                region.paste(mirrored,(start+width*a,0,start+width*(1+a),dim) )



          if bottom< 0:

            overlapbottom = 0 - bottom
            end = min(dim,(2*overlapbottom))
            copy = region.crop((0,overlapbottom,dim,end))
            height = copy.size[1]
            mirrored = copy.transpose(1)
            noverlap = int(overlapbottom/height)
            for a in range(1, noverlap+1):
                region.paste(mirrored,(0,end-height*(a+1),dim,end-height*a) )

          if top> h:

            overlaptop = top - h
            start = max(0,dim-(2*overlaptop))
            copy = region.crop((0,start,dim,(dim-overlaptop)))
            height = copy.size[1]
            mirrored = copy.transpose(1)
            noverlap = int(overlaptop/height)
            for a in range(1, noverlap+1):
                region.paste(mirrored,(0,start+height*a,dim,start+height*(1+a)) )        




                
          with torch.no_grad():
         
                  imagepre = np.array(region)

                  image = imagepre#.astype(np.uint8)
                  image = normaliseimg(normalisationparam,image)
                  transformed = transform1(image=image)
                  image = transformed["image"]

                  img = image.float().cuda()
                 
                  img = torch.unsqueeze(img,0)
                  pred = Net(img)

                

                  pred = pred.type(torch.FloatTensor)
                  

                
                  out = pred.cpu().detach().numpy()
                  output = Image.fromarray(out[0][0])

                
                  outputf = output.resize((dim,dim))
                  picpaste3.paste(outputf, box) 
                  #pic3.paste(outputf, box)
    picnp = np.array(pic)
    pic1np = np.array(picpaste1)
    pic2np = np.array(picpaste2)
    pic3np = np.array(picpaste3)
    picavnp = pic1np/3+pic2np/3+pic3np/3
    
    #npclip = np.clip(pic6np,0,255)
  #  original = Image.open(test)        



 #   originalnp =np.array(original)
#    originalnormnp=(originalnp-originalnp.min())/(originalnp.max()-originalnp.min())
    picnp = (picnp-picnp.min())/(picnp.max()-picnp.min())

    
    picoutnp = 1/(1+np.exp(-picavnp))
    
    #picoutnp = picnormavnp>0.5
   # picoutnp2 = picavnpnorm>0.5
    picout = Image.fromarray((picoutnp * 255).astype(np.uint8))
    picout.save(test+'/results/picout{}{}{}{}.tiff'.format(type1,network,encoder,savetag))
    im2 = picout 
    
   # im = Image.open('test2/testmask.png')
    #im2 = Image.open('test2/results/picout{}.png'.format(type1))
    im2 = ImageOps.grayscale(im2)
    prediction = np.array(im2)
    
    THRESHOLD = 125
    prediction = prediction > THRESHOLD
    prediction = prediction.astype(int)
    
    
    # fig, ax = plt.subplots(1,2,figsize=(10, 20))
    # ax1 = plt.subplot(1,2,1)
    # plt.imshow(picout)
    # ax2 = plt.subplot(1,2,2)
    # plt.imshow(prediction)
    # ax1.title.set_text('output')
    # ax2.title.set_text('output binary')
    
    
    
    im = Image.open(test+testmask)
    target = np.array(im)/255
    target = target>0.1
    target = target.astype(int)

    
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    width, height = im.size
    nopix = width*height
    predictionbin = np.logical_and(prediction, prediction)
    targetbin = np.logical_and(target,target)
    TP  = np.sum(intersection)
    FP = np.sum(predictionbin)-TP
    FN = np.sum(targetbin)-np.sum(intersection)
    precision = TP/(TP+FP)
    Recall = TP/(TP+FN)


    picoutfitnp = blockfit((picoutnp * 255).astype(np.uint8))
    predictionfit = picoutfitnp.astype(int) 
    picoutfit = Image.fromarray((picoutfitnp * 255).astype(np.uint8))
    picoutfit.save(test+'/results/picout{}{}{}{}fit.tiff'.format(type1,network,encoder,savetag))
    intersectionfit = np.logical_and(target, predictionfit)
    unionfit = np.logical_or(target, predictionfit)
    iou_scorefit = np.sum(intersectionfit) / np.sum(unionfit)

    print("IOU = {}".format(iou_score))
    print("precision = {}".format(precision))
    print("recall = {}".format(Recall))
    return iou_score,iou_scorefit,precision,Recall