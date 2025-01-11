import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels

class FaceKeypointResNet50(nn.Module):
    def __init__(self,pretrained, requires_grad):
        super(FaceKeypointResNet50, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__["resnet50"](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__["resnet50"](pretrained=None)

        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
            print('Training intermediate layer parameters...')
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Freezing intermediate layer parameters...')
            
        self.l0 = nn.Linear(2048,136)

    def forward(self, x):
        batch,_,_,_ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(batch,-1)
        l0 =self.l0(x)
        return l0