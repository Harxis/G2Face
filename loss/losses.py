import torch
from torch import nn
from torch.nn import functional as F
import lpips


class ReconLoss(nn.Module):
    def __init__(self, l1_coef=1, lpips_coef=1):
        super(ReconLoss, self).__init__()
        self.l1_coef = l1_coef
        self.lpips_coef = lpips_coef
        self.lpips_criterion = lpips.LPIPS(net='vgg')
        self.lpips_criterion.eval()


    def forward(self, ori_image, recon_image):
        L_l1 = F.l1_loss(ori_image, recon_image)
        L_lpips = self.lpips_criterion(ori_image, recon_image).mean()
        total_loss = self.l1_coef * L_l1 + self.lpips_coef * L_lpips 
        return total_loss, {'recon_loss': total_loss, 'l1_loss': L_l1, 'lpips_loss': L_lpips}
    

class GeometryLoss(nn.Module):
    def __init__(self, shape_coef=1, lm_coef=1, color_coef=1):
        super(GeometryLoss, self).__init__()
        self.shape_coef = shape_coef
        self.lm_coef = lm_coef
        self.color_coef = color_coef

    def forward(self, 
                shape,
                shape_recon,
                landmark,
                landmark_recon,
                color,
                color_recon):

        L_shape = F.mse_loss(shape, shape_recon)
        L_lm = F.mse_loss(landmark, landmark_recon)
        L_color = F.mse_loss(color, color_recon)
        total_loss =  self.shape_coef * L_shape + self.lm_coef * L_lm + self.color_coef * L_color
        return total_loss, {'shape_loss': L_shape.detach(), 'landmark_loss': L_lm.detach(), 'color_loss': L_color.detach()}
    

class IDLoss(nn.Module):
    def __init__(self, id_coef=1):
        super(IDLoss, self).__init__()
        self.id_coef = id_coef
        self.cos_embedding_loss = nn.CosineEmbeddingLoss()

    def forward(self, 
                id_source,
                id_target,
                target):
        L_id = self.cos_embedding_loss(id_source, id_target, target)
        total_loss =  self.id_coef * L_id
        return total_loss, {'id_loss': L_id.detach()}


class IDDiversityLoss(nn.Module):
    def __init__(self, diversity_coef=1) -> None:
        super().__init__()
        self.div_coef = diversity_coef
    
    def forward(self, id):
        cos_sim_matrix = torch.cosine_similarity(id.unsqueeze(1), id.unsqueeze(0), dim=2)
        coe_mat = torch.ones_like(cos_sim_matrix)-torch.eye(cos_sim_matrix.shape[0]).to(cos_sim_matrix.device)
        id_sim = (coe_mat*cos_sim_matrix)
        zero_mat = torch.zeros_like(id_sim)
        L_id_div = torch.where(id_sim<0,zero_mat, id_sim).mean()
        total_loss = self.div_coef * L_id_div
        return total_loss, {'diversity_loss': L_id_div.detach()}


class InfoHidingLoss(nn.Module):
    def __init__(self, hiding_coef=1) -> None:
        super(InfoHidingLoss, self).__init__()
        self.hiding_coef = hiding_coef

    def forward(self, extract_info, hiding_info):
        hiding_loss = F.binary_cross_entropy_with_logits(extract_info, hiding_info)
        total_loss = self.hiding_coef * hiding_loss 
        return total_loss, {'hiding_loss': hiding_loss.detach()}