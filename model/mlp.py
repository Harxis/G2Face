from torch import nn
import torch
import math
from torch.nn import functional as F
from torch.autograd import Function


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5, bias=True):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return scale * F.leaky_relu(input + bias.view((1, -1) + (1,) * (len(input.shape) - 2)),
                                negative_slope=negative_slope)


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)
    

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
        # self.leakyrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # self.prelu = nn.PReLU()

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
            # out = self.leakyrelu(out)
        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )
        return out



class Map2ID(nn.Module):
    def __init__(self, input_dim=512):
        super(Map2ID, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, 512))
        layers.append(nn.LeakyReLU(negative_slope=0.2))
        layers.append(nn.Linear(512, 512))
        layers.append(nn.LeakyReLU(negative_slope=0.2))
        layers.append(nn.BatchNorm1d(512))
        self.map_2_id = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.map_2_id(z)
    
    

class MLP(nn.Module):
    def __init__(self, latent_dim, style_dim, n_mlp, lr_mlp=0.01) -> None:
        super(MLP, self).__init__()
        self.norm = PixelNorm()
        
        layers_low = [EqualLinear(latent_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu")]
        layers_middle = [EqualLinear(latent_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu")]
        layers_high = [EqualLinear(latent_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu")]

        for i in range(n_mlp-1):          
            layers_low.append(
                  EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                  )
                )
            layers_middle.append(
                  EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                  )
                )
            layers_high.append(
                  EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                  )
                )
            
        self.style_low = nn.Sequential(*layers_low)
        self.style_middle = nn.Sequential(*layers_middle)
        self.style_high = nn.Sequential(*layers_high)

    def forward(self, latent_id, latent_shape):      
        latent_blend = torch.cat([latent_id, latent_shape], dim=1)
        latent_blend = self.norm(latent_blend)
        style_low = self.style_low(latent_blend)
        style_middle = self.style_middle(latent_blend)
        style_high = self.style_high(latent_blend)   

        style_low = style_low.unsqueeze(1).repeat(1, 6, 1)
        style_middle = style_middle.unsqueeze(1).repeat(1, 6, 1)
        style_high = style_high.unsqueeze(1).repeat(1, 2, 1)
        style = torch.cat([style_low, style_middle, style_high], dim=1)
        return style



if __name__ =='__main__':
    from torchsummary import summary
    net = MLP(latent_dim=769, style_dim=512, n_mlp=8)
    summary(net, [(512,), (257)])


