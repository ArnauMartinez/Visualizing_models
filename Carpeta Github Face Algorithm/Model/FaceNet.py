

from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.linear import Linear
import torch.nn as nn 




class Inverted_Residual(nn.Module):
    def __init__(self,inp, oup,expand_ratio,stride=1):
        super(Inverted_Residual,self).__init__()
        self.stride=stride
        hidden_dim=round(inp*expand_ratio)
        self.use_res_connect= self.stride==1 and inp==oup 
        assert stride in [1,2]

        if expand_ratio==1:
            self.conv = nn.Sequential(
                #depthwise
                nn.Conv2d(hidden_dim,hidden_dim,3,stride,1,groups=hidden_dim,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #pointwise
                nn.Conv2d(hidden_dim, oup,1,1,0,bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                #pointwise widing
                nn.Conv2d(inp,hidden_dim,1,1,0,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                #depthwise
                nn.Conv2d(hidden_dim,hidden_dim,3,stride,1,groups=hidden_dim,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                #pointwise narrowing
                nn.Conv2d(hidden_dim,oup, 1,1,0, bias=False),
                nn.BatchNorm2d(oup),
            )
    def forward(self,x):
        if self.use_res_connect:
            return self.conv(x)+x
        else:
            return self.conv(x)



def conv_bn(inp,oup,stride):
    return nn.Sequential(
        nn.Conv2d(inp,oup,3,stride,1,bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
        nn.MaxPool2d(2,2),
    )

def conv_1x1_bn(inp,oup,last_lyr):
    if last_lyr:
        return nn.Sequential(
            nn.Conv2d(inp,oup,1,1,0,bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp,oup,1,1,0,bias=False),
            nn.BatchNorm2d(oup)
            
        )




class FaceNet(nn.Module):
    
    def __init__(self, width_mult=1., input_size_x=768, input_size_y=512, last_lyr=True):
        super(FaceNet,self).__init__()
        block=Inverted_Residual
        inp_channel=32
        last_channel=1280
        self.last_channel=int(last_channel*width_mult) if width_mult > 1 else last_channel
        inp_channel=int(inp_channel*width_mult)
        self.last_layer=last_lyr
        inverted_residual_setting=[
            #t,  c,  n,  s
            [1, 16,  1,  1],
            [6, 24,  2,  2],
            [6, 32,  1,  2],
            [6, 64,  3,  2],
            [6, 96,  1,  1],
            [6,160,  2,  2],
            [6,320,  1,  1],
        ]
        

        self.features=[conv_bn(3,inp_channel,2)]

        for t,c,n,s in inverted_residual_setting:
            output_channel= int(c*width_mult)

            for i in range(n):
                if i==0:
                    self.features.append(block(inp_channel,output_channel,stride=s,expand_ratio=t))
                else:
                    self.features.append(block(inp_channel,output_channel,stride=1,expand_ratio=t))
                inp_channel=output_channel
        

        
        #last convolutional layer
        self.features.append(conv_1x1_bn(inp_channel,self.last_channel,last_lyr))
        self.features=nn.Sequential(*self.features)

        if last_lyr:
            self.features2=nn.Sequential(nn.Dropout(0.1), nn.Linear(self.last_channel,725))
        
    
    def forward(self,x):
         x=self.features(x)
         x=x.mean(3).mean(2)
         if self.last_layer:
             x=self.features2(x)
         return x 

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,Conv2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                nn.init.xavier_normal_(m.weight.data)
            if isinstance(m,BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m,BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m,Linear):
                m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()
            
FaceNet()





      


        
