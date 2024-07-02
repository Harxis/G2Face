import torch
from torch import nn
from torch.nn import functional as F

class HidingExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        body = []
        for i in range(3):
            body.append(ExtractorBlock())
        self.body = nn.Sequential(*body)
        self.conv1x1_input = nn.Conv2d(3, 128, kernel_size=1, stride=1, bias=True)
        self.conv1x1_output = nn.Conv2d(128, 1, kernel_size=3, stride=2, padding=1, bias=True)
        self.activate = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.conv1x1_input(x)
        x = self.activate(x)
        x = self.body(x)
        x = self.conv1x1_output(x)
        return x.view(x.size(0), 512, 32)


class ExtractorBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1x1_branch_1 = nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=True)
        self.bn_branch_1_1 = nn.BatchNorm2d(64)
        self.conv3x3_branch_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_branch_1_2 = nn.BatchNorm2d(64)
        self.conv1x1_branch_2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=True)
        self.bn_branch_2_1 = nn.BatchNorm2d(64)
        self.activate = nn.LeakyReLU(negative_slope=0.1)
    
    def forward(self, x):
        x_branch_1 = self.conv1x1_branch_1(x)
        x_branch_1 = self.bn_branch_1_1(x_branch_1)
        x_branch_1 = self.activate(x_branch_1)

        x_branch_1 = self.conv3x3_branch_1(x_branch_1)
        x_branch_1 = self.bn_branch_1_2(x_branch_1)
        x_branch_1 = self.activate(x_branch_1)

        x_branch_2 = self.conv1x1_branch_2(x)
        x_branch_2 = self.bn_branch_2_1(x_branch_2)
        x_branch_2 = self.activate(x_branch_2)

        return torch.cat([x_branch_1, x_branch_2], dim=1)
    

if __name__ == '__main__':
    x = torch.randn(2, 3, 128, 128)
    net = HidingExtractor()
    print(net(x).size())