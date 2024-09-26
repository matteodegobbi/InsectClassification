from torch import nn 
import torch

class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,kernel_size,padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,padding=padding ,stride=1,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.actv = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.actv(x)
        return x

class inceptBlockA(nn.Module):#1x1 convolution 
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = BasicConv2d(in_channels,out_channels,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x



class inceptBlockB(nn.Module):#1x1 convolution + 3x1 convolution
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = BasicConv2d(in_channels,out_channels,1)
        self.conv2 = BasicConv2d(out_channels,out_channels,(3,1),padding=(1,0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x



class inceptBlockC(nn.Module):#1x1 convolution + 5x1 convolution
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = BasicConv2d(in_channels,out_channels,1)
        self.conv2 = BasicConv2d(out_channels,out_channels,(5,1),padding=(2,0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class inceptBlockD(nn.Module):#1x1 convolution + 3x1 pool
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = BasicConv2d(in_channels,out_channels,1)
        self.pool = nn.MaxPool2d((3,1),padding=(1,0),stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool(x)
        return x

class inceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.blockA = inceptBlockA(in_channels,out_channels)
        self.blockB = inceptBlockB(in_channels,out_channels)
        self.blockC = inceptBlockC(in_channels,out_channels)
        self.blockD = inceptBlockD(in_channels,out_channels)
        self.finalConvolution = BasicConv2d(4*out_channels,out_channels,kernel_size=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.blockA(x)
        #print(a.shape)
        b = self.blockB(x)
        #print(b.shape)
        c = self.blockC(x)
        #print(c.shape)
        d = self.blockD(x)
        #print(d.shape)
        out = torch.cat([a,b,c,d],dim=1)
        out = self.finalConvolution(out)

        return out