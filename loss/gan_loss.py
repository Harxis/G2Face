import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

def make_r1_gp(discr_real_pred, real_batch):

    grad_real = torch.autograd.grad(outputs=discr_real_pred.sum(), inputs=real_batch, create_graph=True, retain_graph=True)[0]
    grad_penalty = grad_real.reshape(grad_real.shape[0], -1).pow(2).sum(dim=1).mean()

    return grad_penalty


class DiscriminatorLoss(nn.Module):
    def __init__(self, gp_coef=5) -> None:
        super(DiscriminatorLoss, self).__init__()
        self.gp_coef = gp_coef
    
    def forward(self, real_batch, discr_real_pred, discr_fake_pred, d_regularize):
        real_loss = F.softplus(-discr_real_pred).mean()
        fake_loss = F.softplus(discr_fake_pred).mean()
        if d_regularize:
            grad_penalty = make_r1_gp(discr_real_pred, real_batch)
        else:
            grad_penalty = torch.zeros_like(real_loss)
        d_loss = real_loss + fake_loss + self.gp_coef * grad_penalty

        return d_loss, {'disc_loss': d_loss.detach(), 'real_loss': real_loss.detach(), 'fake_loss': fake_loss.detach(), 'grad_penalty_loss': grad_penalty.detach()}
        

class GeneratorLoss(nn.Module):
    def __init__(self) -> None:
        super(GeneratorLoss, self).__init__()

    def forward(self, discr_fake_pred):
        gen_loss = F.softplus(-discr_fake_pred).mean()
        return gen_loss, {'gen_loss': gen_loss.detach()}
