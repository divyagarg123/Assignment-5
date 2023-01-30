import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Model(nn.Module):
    def __init__(self, group_norm, layer_norm):
        super(Model, self).__init__()
        self.group_norm = group_norm
        self.layer_norm = layer_norm
        # Inputblock
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3,3), padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout(0.05)
        )

        if self.group_norm:
            self.norm1= nn.GroupNorm(2,10)
        elif self.layer_norm:
            self.norm1=nn.GroupNorm(1,10)
        else:
            self.norm1=nn.BatchNorm2d(10)

        # Conv block 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3,3), padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout(0.05)
        )

        if self.group_norm:
            self.norm2= nn.GroupNorm(2,10)
        elif self.layer_norm:
            self.norm2=nn.GroupNorm(1,10)
        else:
            self.norm2=nn.BatchNorm2d(10)

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3,3), padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout(0.05)
        )

        if self.group_norm:
            self.norm3= nn.GroupNorm(4,20)
        elif self.layer_norm:
            self.norm3=nn.GroupNorm(1,20)
        else:
            self.norm3=nn.BatchNorm2d(20)

        #Trans Block1
        self.pool1 = nn.MaxPool2d(2,2)
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout(0.05)
        )

        if self.group_norm:
            self.norm4= nn.GroupNorm(2,10)
        elif self.layer_norm:
            self.norm4=nn.GroupNorm(1,10)
        else:
            self.norm4=nn.BatchNorm2d(10)

        # Conv block 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3,3), padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout(0.05)
        )

        if self.group_norm:
            self.norm5= nn.GroupNorm(2,10)
        elif self.layer_norm:
            self.norm5=nn.GroupNorm(1,10)
        else:
            self.norm5=nn.BatchNorm2d(10)

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3,3), padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout(0.05)
        )

        if self.group_norm:
            self.norm6= nn.GroupNorm(4,20)
        elif self.layer_norm:
            self.norm6=nn.GroupNorm(1,20)
        else:
            self.norm6=nn.BatchNorm2d(20)

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) # output_size = 7
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        )
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 1

    def forward(self, x):
        x = self.convblock1(x)
        x = self.norm1(x)
        x = self.convblock2(x)
        x = self.norm2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.norm3(x)
        x = self.convblock4(x)
        x = self.norm4(x)
        x = self.convblock5(x)
        x = self.norm5(x)
        x = self.convblock6(x)
        x = self.norm6(x)
        x = self.convblock7(x)
        x=self.gap(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
