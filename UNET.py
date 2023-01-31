# INPUTS
#test = path to data folder
#inno = number of input channels
#batch = number of batches
#path = path to datafolder
#nepochs = max number of training epochs (note: training is set to end once training IOU is 15% less than test IOU)
#network = type of network eg. Unet
#encoder = name of encoder eg. mobilenet_v2
#pretrain = dataset for pretraining weights subject to availability on the segmentation pytorch toolbox. eg. imagenet
#dim = dimension of image crops used in network. Images will be square. eg. dim = 512 will mean 512x512 patches.
#transparams = data augmentation (albumentations) to use for training
#transparams = data augmentation (albumentations) to use for validation

# INPUT FILES
# 

#OUTPUTS
# iou_score = Intersection over union score for test image
# precision = precisions score for test image
# recall = recall score for test image
#nepochsf = number of epochs used for training (note: number is appended to name of trained network)

def unet(test, inno, batch, path, nepochs, network, encoder, pretrain, dim,transparams, transparamsv):

    import os
    import sys
    from RUN import UNETrun
  #  path1 = 'C:/Users/eejmws/OneDrive - University of Leeds/Pytorch/'
  #  path2 = 'C:/Users/jackm/OneDrive - University of Leeds/Pytorch/'

  #  isdir1 = os.path.isdir(path1) 
  #  isdir2 = os.path.isdir(path2) 

  #  if isdir1 == True:
  #      os.chdir(path1)
  #  elif isdir2 == True:
  #      os.chdir(path2)
  #  else:
  #      print("directory not found")
    results_path = path+"/results"
    try:
        os.makedirs(results_path)
    except OSError:
        pass    
    import numpy as np
    import torch
    import segmentation_models_pytorch as smp
    import torchvision
    from torch import nn, optim
    from torchvision import datasets
    import matplotlib.pyplot as plt
    import cv2
    import math
    from PIL import Image
    from skimage.io import imread
    from torch.utils import data
    import albumentations as A
    from torch.utils.data import Dataset as BaseDataset
    import torchvision.transforms as transforms
    #from transformations import ComposeDouble, FunctionWrapperDouble, create_dense_target, normalize_01
    #from customdatasets import SegmentationDataSet1
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split
    import pathlib
    import time
    from albumentations.pytorch import ToTensorV2
    DATA_DIR = './test2/'
    # First check devices available (gpus or cpu), 'cuda' stands for gpus
    if torch.cuda.is_available(): device = torch.device('cuda')
    else: device = torch.device('cpu')
    
    print(device)
    transform1 = A.Compose(transparams)
    transform2 = A.Compose(transparamsv)
    trans = A.Compose([
            ToTensorV2()])
    class MasonryDataset(BaseDataset):

        CLASSES = ['masonry', 'mortar']

        def __init__(self,
                     inputs: list,
                     targets: list,
                     transform=None
                     ):
            self.inputs = inputs
            self.targets = targets
            self.transform = transform
            self.inputs_dtype = torch.float32
            self.targets_dtype = torch.float32

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self,
                        index: int):
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            permissionerr = 1
            while permissionerr == 1:
                try:
                    x, y = imread(input_ID), imread(target_ID)
                    permissionerr = 0
                except:
                    permissionerr = 1
                    print("permision error!!! retrying...")
                    time.sleep(1)
            (x-x.mean())/np.std(x)    
            maxinput = x.max()
            x = x/maxinput
            if y.max() != 0:
                y = y/y.max()
            # Preprocessing

            if self.transform is not None:
                transformed1 = self.transform(image=x, mask=y)
                x = transformed1['image']
                y = transformed1['mask']
            transformed = trans(image=x, mask=y)
            x = transformed['image']
            x=x*maxinput
            y = transformed['mask']
            y = torch.unsqueeze(y,0)
            y= y.float()
            # Typecasting
           # x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
           # y = y.view(1, y.shape[0], y.shape[1])
            #x = x.view(1, x.shape[0], x.shape[1])
           # y = torch.from_numpy(y).type(self.targets_dtype)
           # sptransform = transforms.Compose([
           # torchvision.transforms.Normalize(
          #  mean=[0.5],
           # std=[0.5],
          #  ),
          #  ])

          #  y = sptransform(y)
           # y = y.view(1, y.shape[0], y.shape[1])
           # y=torch.squeeze(y)
           # y = y.long()
            return x, y


    def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
        """Returns a list of files in a directory/path. Uses pathlib."""
        filenames = [file for file in path.glob(ext) if file.is_file()]
        return filenames

    root = pathlib.Path.cwd()
    # input and target files
    inputs = get_filenames_of_path(root / (path+"/croppedimages"))
    targets = get_filenames_of_path(root / (path+"/croppedmasks"))

    # random seed
    random_seed = 42

    # split dataset into training set and validation set
    train_size = 0.8  # 80:20 split

    inputs_train, inputs_valid = train_test_split(
        inputs,
        random_state=random_seed,
        train_size=train_size,
        shuffle=True)

    targets_train, targets_valid = train_test_split(
        targets,
        random_state=random_seed,
        train_size=train_size,
        shuffle=True)

    # dataset training
    dataset_train = MasonryDataset(inputs=inputs_train,
                                        targets=targets_train,
                                        transform=transform1)

    # dataset validation
    dataset_valid = MasonryDataset(inputs=inputs_valid,
                                    targets=targets_valid,
                                    transform=transform2)

    # dataloader training
    dataloader_training = DataLoader(dataset=dataset_train,
                                     batch_size=batch,
                                     shuffle=True)

    # dataloader validation
    dataloader_validation = DataLoader(dataset=dataset_valid,
                                       batch_size=batch,
                                       shuffle=True)
    net = getattr(smp, network)(
        encoder_name= encoder,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=pretrain,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=inno,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
    ) 
    net.to("cuda")
    def train():


        def stats(loader, net):

            #loss_fn = nn.CrossEntropyLoss()
            correct = 0
            total = 0
            running_loss = 0
            n = 0    # counter for number of minibatches
            with torch.no_grad():
                for data in loader:
                    images, masks = data
                    images, masks = images.to("cuda"), masks.to("cuda")
                    outputs = net(images)      

                    # accumulate loss
                    running_loss += criterion(outputs, masks).item()
                    n += 1

                    # accumulate data for accuracy
                  #  _, predicted = torch.max(outputs.data, 1)
                  #  total += labels.size(0)    # add in the number of labels in this minibatch
                    #correct += (predicted == labels).sum().item()  # add in the number of correct labels


            return running_loss/n

        
        #results_path = './results_UNET_mobilenet_RGB200.pt'
        statsrec = np.zeros((2,nepochs))

       # loss_fn = nn.CrossEntropyLoss()
        #optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.01)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001, eps=1e-08,  weight_decay=0.00001)


        criterion = nn.BCEWithLogitsLoss(pos_weight=10*torch.ones([1])).to("cuda")
        for epoch in range(nepochs):  # loop over the dataset multiple times
            correct = 0          # number of examples predicted correctly (for accuracy)
            total = 0            # number of examples
            running_loss = 0.0   # accumulated loss (for mean loss)
            n = 0                # number of minibatches
            
            
            
            for data in dataloader_training:
                images, masks = data
                images, masks = images.to("cuda"), masks.to("cuda")

                 # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward, backward, and update parameters
                outputs1 = net(images)
                outputs = net(outputs1)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                # accumulate loss
                running_loss += loss.item()
                n += 1

                # accumulate data for accuracy
             #   _, predicted = torch.max(outputs.data, 1)
             #   total += labels.size(0)    # add in the number of labels in this minibatch
             #   correct += (predicted == labels).sum().item()  # add in the number of correct labels

            # collect together statistics for this epoch
            ltrn = running_loss/n
         #   atrn = correct/total 
            ltst = stats(dataloader_validation, net)



            print(f"epoch: {epoch} training loss: {ltrn: .3f}   test loss: {ltst: .3f} ")
            nepochsf = epoch

       #     print(ltrn)
        #    print(atrn)
         #   print(ltst)
          #  print(atst)
           # print(epoch)
            #print(statsrec)
           # print(ltrn.type())
            #print(atrn.type())
            #print(ltst.type())
            #print(atst.type())
            #print(epoch.type())
            #print(statsrec.type())
            statsrec[:,epoch] = (ltrn, ltst)
            
          #  if epoch >= 10:
          #      if 0.85*ltst >= ltrn:
          #          break
            
        # save network parameters, losses and accuracy
        savestate = '/Final_epoch_{}{}{}.pt'.format(nepochsf, network, encoder)
        print(savestate)
        saveplace = results_path+savestate
        torch.save({"state_dict": net.state_dict(), "stats": statsrec}, saveplace)
        if test == 1:    
               iou_score, precision,recall = UNETrun(inno, savestate, "depth", network, path, encoder, dim, "/test.tiff","/testmask.tiff", "default")
        else:
               iou_score, precision,recall = [0,0,0]
        return iou_score, precision,recall, nepochsf
    
    
    #if __name__ == '__main__':
    iou_score, precision, recall,nepochsf = train()
    torch.cuda.empty_cache()
 #   torch.save({"state_dict" : net.state_dict()}, results_path)
    return iou_score,precision,recall, nepochsf