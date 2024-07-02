import argparse
from xmlrpc.client import Boolean
import torch
from torch import nn
import numpy as np
from . import networks
from .bfm import ParametricFaceModel
from .renderer import MeshRenderer
from .utils_3dmm import util
from .base_model import BaseModel



class FaceReconModel(BaseModel):
    def __init__(self, opt) -> None:
        BaseModel.__init__(self, opt)

        self.opt = opt
        self.visual_names = ['output_vis']
        self.model_names = ['net_recon']
        self.parallel_names = self.model_names + ['renderer']

        self.net_recon = networks.define_net_recon(
            net_recon=opt.net_recon, use_last_fc=opt.use_last_fc, init_path=opt.init_path
        )
        self.face_model= ParametricFaceModel(
            bfm_folder=opt.bfm_folder, camera_distance=opt.camera_d, focal=opt.focal, center=opt.center,
            is_train=False, default_name=opt.bfm_model
        )
        fov = 2 * np.arctan(opt.center / opt.focal) * 180 / np.pi
        self.renderer = MeshRenderer(
            rasterize_fov=fov, znear=opt.z_near, zfar=opt.z_far, rasterize_size=int(2 * opt.center), use_opengl=opt.use_opengl
        )

    def to(self, device):
        self.device = torch.device(device)
        self.net_recon.to(self.device)
        self.face_model.to(self.device)
        self.renderer.to(self.device)

    def compute_coeff(self, x):
        output_coeff = self.net_recon(x)
        pred_coeffs_dict = self.face_model.split_coeff(output_coeff)
        return output_coeff, pred_coeffs_dict

    def compute_3d_info(self, coeff):
        pred_vertex, pred_tex, pred_color, pred_mesh, pred_lm = self.face_model.compute_for_render(coeff)
        return pred_vertex, pred_tex, pred_color, pred_mesh, pred_lm

    def compute_3d_shape(self, coeff):
        face_shape, landmark, color = self.face_model.compute_for_loss(coeff)
        return face_shape, landmark, color

    def reconstruct_face(self, vertex, feature):
        pred_mask, _, pred_face = self.renderer(vertex=vertex, tri=self.face_model.face_buf, feat=feature)
        return pred_mask, pred_face

    def compute_visuals(self, x, pred_face, pred_mask, pred_lm):
        with torch.no_grad():
            input_img_numpy = 255. * x.detach().cpu().permute(0, 2, 3, 1).numpy()
            output_vis = pred_face * pred_mask + (1 - pred_mask) * x
            output_vis_numpy_raw = 255. * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()
            
            pred_lm_numpy = pred_lm.detach().cpu().numpy()
            output_vis_numpy = util.draw_landmarks(output_vis_numpy_raw, pred_lm_numpy, 'r')
            output_vis_numpy = np.concatenate((input_img_numpy, 
                                output_vis_numpy_raw, output_vis_numpy), axis=-2)

            output_vis = torch.tensor(
                    output_vis_numpy / 255., dtype=torch.float32
                ).permute(0, 3, 1, 2).to(self.device)
        
        return output_vis


if __name__ == '__main__':
    from utils.visualizer import MyVisualizer
    from utils.preprocess import align_img
    from utils.load_mats import load_lm3d
    import os
    from collections import OrderedDict
    from PIL import Image
    from torchvision.utils import save_image


    def modify_commandline_options():
        """  Configures options specific for CUT model
        """
        parser = argparse.ArgumentParser()
        # net structure and parameters
        parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='network structure')
        parser.add_argument('--init_path', type=str, default='checkpoints/ResNet50/epoch_20.pth')
        parser.add_argument('--use_last_fc', type=bool, default=False, help='zero initialize the last fc')
        parser.add_argument('--bfm_folder', type=str, default='BFM')
        parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

        # renderer parameters
        parser.add_argument('--focal', type=float, default=1015.)
        parser.add_argument('--center', type=float, default=112.)
        parser.add_argument('--camera_d', type=float, default=10.)
        parser.add_argument('--z_near', type=float, default=5.)
        parser.add_argument('--z_far', type=float, default=15.)
        parser.add_argument('--use_opengl', type=bool, default=True, help='use opengl context or not')

        parser.add_argument('--name', type=str, default='face_recon', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='models are saved here')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')

        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--dataset_mode', type=str, default=None, help='chooses how datasets are loaded. [None | flist]')
        parser.add_argument('--img_folder', type=str, default='examples', help='folder for test images.')

        opt, _ = parser.parse_known_args()
        parser.set_defaults(
                focal=1015., center=112., camera_d=10., use_last_fc=False, z_near=5., z_far=15.
            )
        args = parser.parse_args()
        return args

    args = modify_commandline_options()

    visualizer = MyVisualizer(args)
    net = FaceReconModel(args)
    net.eval()
    def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
        # to RGB 
        im = Image.open(im_path).convert('RGB')
        W,H = im.size
        lm = np.loadtxt(lm_path).astype(np.float32)
        lm = lm.reshape([-1, 2])
        lm[:, -1] = H - 1 - lm[:, -1]
        _, im, lm, _ = align_img(im, lm, lm3d_std)
        if to_tensor:
            im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            lm = torch.tensor(lm).unsqueeze(0)
        return im, lm

    lm3d_std = load_lm3d(args.bfm_folder) 
    x, _ = read_data('000002.jpg', '000002.txt', lm3d_std)
    save_image(x, '.3d.png')
    x = x.cuda()
    
    net.device = x.device
    output_coeff, _ = net.compute_coeff(x)
    pred_vertex, pred_tex, pred_color, pred_lm = net.compute_3d_info(output_coeff)
    pred_mask, pred_face = net.reconstruct_face(pred_vertex, net.face_model.face_buf, pred_color)
    visuals = net.compute_visuals(x, pred_face=pred_face, pred_mask=pred_mask, pred_lm=pred_lm)
    visual_ret = OrderedDict()
    visual_ret['a'] = visuals
    visualizer.display_current_results(visual_ret, 0, 20, dataset='ResNet50'.split(os.path.sep)[-1], save_results=True, count=0, name='02', add_image=False)
