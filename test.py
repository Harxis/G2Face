import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import sys
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import random
from torchvision import datasets, transforms
import numpy as np
from torchvision.utils import save_image

from model import HidingExtractor, iresnet50, MLP, FaceReconModel, Generator, Map2ID
from option import options
from utils.util import get_normalizer
from utils.dataset import ImageDataset
from utils.binary_converter import float2bit, bit2float


def image_transform():
    data_transform = transforms.Compose([transforms.Resize((256, 256)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    return data_transform


def collate_pil(x): 
    out_x, out_y = [], [] 
    for xx, yy in x: 
        out_x.append(xx) 
        out_y.append(yy) 
    return out_x, out_y 


def process_image(opt, image_path, save_path, checkpoint_path, batch_size):
    res = ['ori', 'rec', 'any']
    for n in res:
        os.makedirs(os.path.join(save_path, '%s' % n), exist_ok=True)

    device = torch.device(opt.device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # model
    anonymous_net = Generator(size=opt.image_size, style_dim=opt.style_dim).to(device)
    anonymous_net.load_state_dict(checkpoint['anonymous_net'])
    face_recon_net = FaceReconModel(opt)
    face_recon_net.to(device)
    face_recon_net.load_networks(20)
    arc_face = iresnet50().to(device)
    arc_face.load_state_dict(torch.load('pretrain/ms1mv3_arcface_r50.pth', map_location='cpu'))
    map_2_id = Map2ID().to(device)
    map_2_id.load_state_dict(checkpoint['map_2_id'])
    style_mlp = MLP(latent_dim=opt.latent_dim, style_dim=opt.style_dim, n_mlp=opt.n_mlp).to(device)
    style_mlp.load_state_dict(checkpoint['style_mlp'])
    hiding_extractor = HidingExtractor().to(device)
    hiding_extractor.load_state_dict(checkpoint['hiding_extractor'])

    dst = ImageDataset(image_path, transform=image_transform())
    
    map_2_id.eval()
    style_mlp.eval()
    hiding_extractor.eval()
    anonymous_net.eval()
    arc_face.eval()
    face_recon_net.eval()

    loader = DataLoader(dst, num_workers=0, batch_size=batch_size, shuffle=False, drop_last=True)

    channel_mean = torch.tensor([0.5, 0.5, 0.5]).to(device)
    channel_std = torch.tensor([0.5, 0.5, 0.5]).to(device)
    normalizer, denormalizer = get_normalizer(channel_mean, channel_std)

    with torch.no_grad():
        batch = 0
        orig_imgs = []
        any_imgs = []
        rec_imgs = []
        any_latent_sim = 0
        rec_latent_sim = 0
        mae = 0
        bit_correct = 0

        for imgs, path in tqdm(loader):
            imgs = imgs.to(device)
            latent_face = arc_face(imgs)
            denorm_imgs = denormalizer(imgs)
            latent_3d, _ = face_recon_net.compute_coeff(denorm_imgs)

            latent_hiding = float2bit(latent_face)
            rand_z = torch.randn([batch_size, 512]).to(device)
            rand_id = map_2_id(rand_z)
            latent_control = style_mlp(rand_id, torch.cat([latent_3d[:, :144], latent_3d[:, 224:227]], dim=1))

            anonymous_image = anonymous_net(imgs, latent_control, latent_hiding)
            latent_any_face = arc_face(anonymous_image)
            extract_info_bin = hiding_extractor(anonymous_image)

            extract_id_float = bit2float(torch.round(torch.sigmoid(extract_info_bin).cpu())).cuda()
            extract_id_float = torch.where(extract_id_float<-50, torch.full_like(extract_id_float, 0.0), extract_id_float)
            extract_id_float = torch.where(extract_id_float>50, torch.full_like(extract_id_float, 0.0), extract_id_float)

            denorm_anonymous_image = denormalizer(anonymous_image)
            latent_3d_anonymous, _ = face_recon_net.compute_coeff(denorm_anonymous_image)
            latent_control_recover = style_mlp(extract_id_float, torch.cat([latent_3d_anonymous[:, 0:144], latent_3d_anonymous[:, 224:227]], dim=1))
            recover_image = anonymous_net(anonymous_image, latent_control_recover, latent_hiding)
            latent_rec_face = arc_face(recover_image)

            any_latent_sim += torch.cosine_similarity(latent_face, latent_any_face, dim=1).mean()
            rec_latent_sim += torch.cosine_similarity(latent_face, latent_rec_face, dim=1).mean()
            mae += F.l1_loss(recover_image, imgs)
            extract_info_bin = torch.round(torch.sigmoid(extract_info_bin))
            bit_correct += ((extract_info_bin.eq(latent_hiding.data)).sum()) / (batch_size * 512 * 32)

            anonymous_image = anonymous_image.clip(-1, 1)
            recover_image = recover_image.clip(-1, 1)
            batch += 1

            orig_imgs.append(imgs.clone().cpu())
            any_imgs.append(anonymous_image.clone().cpu())
            rec_imgs.append(recover_image.clone().cpu())

            for i in range(anonymous_image.size(0)):
                save_file_name = path[i]
                original = imgs[i]
                anonymous = anonymous_image[i]
                recover = recover_image[i]
                original = original.unsqueeze(0)
                anonymous = anonymous.unsqueeze(0)
                recover = recover.unsqueeze(0)
                save_image(original, os.path.join(save_path, 'ori', save_file_name), normalize=True, scale_each=True)
                save_image(anonymous, os.path.join(save_path, 'any', save_file_name), normalize=True, scale_each=True)
                save_image(recover, os.path.join(save_path, 'rec', save_file_name), normalize=True, scale_each=True)

            if batch > 100:
                break

        any_latent_sim = any_latent_sim / batch
        rec_latent_sim = rec_latent_sim / batch
        mae = mae / batch
        bit_correct = bit_correct / batch

        print(f"Anonymous id similarity: {any_latent_sim:.4f} \
                Recover id similarity: {rec_latent_sim:.4f} \
                Recover image MAE: {mae:.4f} \
                Bit error rate: {1-bit_correct:.4f}")
        saved_imgs = []
        for res1, res2, res3 in zip(orig_imgs[:10], any_imgs[:10], rec_imgs[:10]):
            saved_imgs.append(torch.cat([res1, res2, res3], dim=0))
        saved_imgs = torch.cat(saved_imgs, dim=0)
        save_image(saved_imgs, os.path.join(save_path, 'result.jpg'), normalize=True, scale_each=True)



if __name__ == '__main__':
    opt = options()
    process_image(opt=opt,
                  image_path=opt.celebahq_path,
                  save_path='test_images/results', 
                  checkpoint_path='weights/G2Face.pth', 
                  batch_size=8)