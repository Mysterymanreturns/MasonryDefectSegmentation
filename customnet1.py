import torch
import numpy as np
import torch.nn as nn

class customnet1(nn.Module):

      def __init__(self):
            super(customnet1, self).__init__()
            net1 = nn.Sequential(
                nn.Conv2d(in_channels=1,out_channels=8, kernel_size=5),    # no padding, stride=1, dilation=1 by default
                # Hout = Hin +1 - kernelsize
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Hout = (Hin -2)/2 + 1
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding = 1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(4608,64*64),     # with 32x32 input, the feature map size reduces to 5x5 with 16 channels.
                nn.ReLU(),
                # nn.Linear(512,512),
                # nn.ReLU(),
                # nn.Linear(512,64*64), 
                # nn.ReLU(),
                nn.Linear(64*64,64*64),
               #nn.Linear(128*128,256*256), 
               # nn.Unflatten(1,(8,4,4)),
               # nn.Unflatten(1,(1,64,64)),
                nn.Unflatten(1,(1,64,64)),
                nn.ConvTranspose2d(1,2,5, stride = 2),
                nn.ReLU(),
                nn.ConvTranspose2d(2,4,3, stride = 1),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.ReLU(),
                nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5),
                nn.ReLU(),
                nn.Conv2d(in_channels=4, out_channels=2, kernel_size=9, padding = 2),
                nn.ReLU(),
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3),

#                 nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReLU(),
#                 nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding = 1),
#                 nn.ReLU(),
              #  nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, padding = 1),
              #  nn.ReLU(),

                #nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),

               # nn.Linear(64,256),
               # nn.Linear(256,256*256)
            )
            self.network1 = net1


      def forward(self,x):
            x = self.network1(x)

            return x

        
class customnet2(nn.Module):

      def __init__(self):
            super(customnet2, self).__init__()
            net1 = nn.Sequential(
                nn.Conv2d(in_channels=1,out_channels=4, kernel_size=5),    # no padding, stride=1, dilation=1 by default
                # Hout = Hin +1 - kernelsize
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Hout = (Hin -2)/2 + 1
                nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5),
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=16, kernel_size=5,),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5),
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(968,2888),
                nn.ReLU(),
                nn.Linear(2888,4096),     # with 32x32 input, the feature map size reduces to 5x5 with 16 channels.
  #              nn.Linear(5000,5000),     # with 32x32 input, the feature map size reduces to 5x5 with 16 channels.
                nn.ReLU(),
                nn.Linear(4096,128*128),

                nn.Unflatten(1,(1,128,128)),
#                # nn.Upsample(scale_factor=2, mode='bilinear'),
#                 nn.ReLU(),
                #nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding = 1),

#                 nn.Upsample(scale_factor=2, mode='nearest'),
              #  nn.ReLU(),
              #  nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding = 1),
              #  nn.ReLU(),
                
                
                
              #  nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, padding = 1),
              #  nn.ReLU(),

                #nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),

               # nn.Linear(64,256),
               # nn.Linear(256,256*256)
               
            )
            self.network1 = net1


      def forward(self,x):
            x = self.network1(x)

            return x
        