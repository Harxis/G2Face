import os
import time
import random
import numpy as np
from itertools import chain
import torch
torch.backends.cudnn.benchmark = True
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.optim.lr_scheduler import ExponentialLR

from torch.utils.tensorboard import SummaryWriter

from utils.distributed import synchronize, get_rank, reduce_loss_dict
from model import HidingExtractor, iresnet50, MLP, FaceReconModel, Generator, Discriminator, augment, AdaptiveAugment, Map2ID
from utils.operate import setup_data
from loss import GeneratorLoss, DiscriminatorLoss, ReconLoss, InfoHidingLoss, IDLoss, GeometryLoss, IDDiversityLoss
from option import options 
from utils.logger import setup_logger
from utils.util import get_normalizer
from utils.binary_converter import float2bit, bit2float
from utils.warmup_scheduler import GradualWarmupScheduler


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        if not 'norm' in k:
            par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def set_model_requires_grad(model, requires_grad, distributed):
    """Sets the `requires_grad` configuration for a particular model."""
    if distributed:
        for param in model.module.parameters():
            param.requires_grad = requires_grad
    else:
        for param in model.parameters():
            param.requires_grad = requires_grad


def train(opt, logger, work_path):
    # ########## Print Setting ############
    if get_rank() == 0:
        logger.info('-' * 50 + 'setting' + '-' * 50)
        for k in opt.__dict__:
            logger.info(k + ": " + str(opt.__dict__[k]))
        logger.info('-' * 107)

    ############# Device ################
    device = torch.device(opt.device)
    
    ########## Set Seeds ###########
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    ######### Model ###########
    anonymous_net = Generator(size=opt.image_size, style_dim=opt.style_dim).to(device)
    state_dict = torch.load('pretrain/550000.pt', map_location='cpu')
    anonymous_net.load_state_dict(state_dict['g_ema'], strict=False)
    face_recon_net = FaceReconModel(opt)
    face_recon_net.to(device)
    face_recon_net.load_networks(20)
    arc_face = iresnet50(return_conv_feature=False).to(device)
    arc_face.load_state_dict(torch.load('pretrain/ms1mv3_arcface_r50.pth', map_location='cpu'))
    map_2_id = Map2ID().to(device)
    style_mlp = MLP(latent_dim=opt.latent_dim, style_dim=opt.style_dim, n_mlp=opt.n_mlp).to(device) # 25235 659
    hiding_extractor = HidingExtractor().to(device)
    discriminator = Discriminator(size=256).to(device)

    ######### Optimizer ###########
    train_params_id = list(map(id, anonymous_net.pyramid.parameters()))
    train_params_id.extend(list(map(id, anonymous_net.fusion_mask.parameters())))
    train_params_id.extend(list(map(id, anonymous_net.hiding_injects.parameters())))
    
    fine_tune_params = filter(lambda p: id(p) not in train_params_id, anonymous_net.parameters())
    train_params = filter(lambda p: id(p) in train_params_id, anonymous_net.parameters())

    train_parameters = chain(train_params,
                             style_mlp.parameters(),
                             map_2_id.parameters(),
                             hiding_extractor.parameters())

    optimizer = optim.Adam([{'params': train_parameters},
                            {'params': fine_tune_params, 'lr': opt.lr*0.1}], 
                            lr=opt.lr, 
                            betas=(0.0, 0.999))
    
    optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.0, 0.999))

    scheduler_explr = ExponentialLR(optimizer, gamma=0.95)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=scheduler_explr)
    optimizer.zero_grad()
    optimizer.step()
    
    start_epoch = 0
    start_iter = 0

    ######### Resume ########### 
    if opt.resume: 
        resume_checkpoint = torch.load(opt.resume_weights, map_location='cpu')
        start_epoch = resume_checkpoint['epoch']
        start_iter = resume_checkpoint['iter']
        anonymous_net.load_state_dict(resume_checkpoint['anonymous_net'])
        style_mlp.load_state_dict(resume_checkpoint['style_mlp'])
        map_2_id.load_state_dict(resume_checkpoint['map_2_id'])
        hiding_extractor.load_state_dict(resume_checkpoint['hiding_extractor'])
        discriminator.load_state_dict(resume_checkpoint['discriminator'])
        optimizer.load_state_dict(resume_checkpoint['optimizer'])
        optimizer_D.load_state_dict(resume_checkpoint['optimizer_D'])
        scheduler_warmup.load_state_dict(resume_checkpoint['scheduler'])
        if get_rank() == 0:
            logger.info('Loading checkpoint from epoch %d.' % start_epoch)

    ######### Distributed DDP ########### 
    if opt.distributed:
        anonymous_net = nn.parallel.DistributedDataParallel(
            anonymous_net,
            device_ids=None,
            output_device=None,
            broadcast_buffers=False,
        )

        face_recon_net.parallelize()

        style_mlp = nn.parallel.DistributedDataParallel(
            style_mlp,
            device_ids=None,
            output_device=None,
            broadcast_buffers=False,
        )
        
        map_2_id = nn.parallel.DistributedDataParallel(
            map_2_id,
            device_ids=None,
            output_device=None,
            broadcast_buffers=False,
        )

        hiding_extractor = nn.parallel.DistributedDataParallel(
            hiding_extractor,
            device_ids=None,
            output_device=None,
            broadcast_buffers=False,
        )
        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=None,
            output_device=None,
            broadcast_buffers=False,
        )

    ######### Loss ###########
    criterion_geometry = GeometryLoss(opt.vert_coef, opt.lm_coef, opt.col_coef).to(device)
    criterion_id = IDLoss(opt.id_coef).to(device)
    criterion_recon = ReconLoss(opt.l1_coef, opt.lpips_coef).to(device)
    criterion_info_hiding = InfoHidingLoss(opt.hiding_coef).to(device)
    criterion_generator = GeneratorLoss().to(device)
    criterion_discriminator = DiscriminatorLoss(opt.gp_coef).to(device)
    criterion_id_diversity = IDDiversityLoss(opt.diversity_coef).to(device)

    ######### DataLoader ###########
    train_loader, test_loader = setup_data(opt)

    ######### Normalizer ###########
    channel_mean = torch.tensor([0.5, 0.5, 0.5]).to(device)
    channel_std = torch.tensor([0.5, 0.5, 0.5]).to(device)
    _, denormalizer = get_normalizer(channel_mean, channel_std)

    #### tensorboard writer ##########
    if get_rank() == 0:
        writer = SummaryWriter(log_dir=work_path)

    #### augment ##############
    ada_aug_p = opt.augment_p if opt.augment_p > 0 else 0.0
    if opt.augment and opt.augment_p == 0:
        ada_augment = AdaptiveAugment(opt.ada_target, opt.ada_length, 8, device)

    iteration = start_iter
    epoch = start_epoch

    arc_face.eval()
    face_recon_net.eval()
    time.sleep(10)

    while True:
        if iteration > opt.num_iter:
            break
        anonymous_process_loss_dict = {}
        reconstruct_process_loss_dict = {}
        GAN_loss_dict = {}
        epoch += 1
        
        id_dummy_div_loss = 0
        id_gen_consist_loss = 0
         
        anonymous_process_l1_loss = 0
        anonymous_process_lpips_loss = 0
        anonymous_process_id_loss = 0
        anonymous_process_shape_loss = 0
        anonymous_process_landmark_loss = 0
        anonymous_process_color_loss = 0
        anonymous_process_hiding_loss = 0
        anonymous_process_gen_loss = 0

        reconstruct_process_l1_loss = 0
        reconstruct_process_lpips_loss = 0
        reconstruct_process_id_loss = 0
        reconstruct_process_shape_loss = 0
        reconstruct_process_landmark_loss = 0
        reconstruct_process_color_loss = 0
        reconstruct_process_hiding_loss = 0
        reconstruct_process_gen_loss = 0

        disc_loss_epoch = 0
        disc_real_loss = 0
        disc_fake_loss = 0
        disc_gp_loss = 0

        ID_same_label = torch.ones([opt.batch_size]).to(device)
        ID_diff_label = torch.full_like(ID_same_label, -1)

        scheduler_warmup.step()
        cur_lr = scheduler_warmup.get_lr()

        map_2_id.train()
        style_mlp.train()
        hiding_extractor.train()

        for i, data in enumerate(train_loader, 0):
            if iteration > opt.num_iter:
                if get_rank() == 0:
                    print('Training %d iteration are finished! Break.' % opt.num_iter)
                break

            iteration += 1
            ori_image = data.cuda()
            real_image_D = ori_image.detach().clone().requires_grad_()
            ######## model mode ########
            set_model_requires_grad(anonymous_net, requires_grad=True, distributed=opt.distributed)
            set_model_requires_grad(style_mlp, requires_grad=True, distributed=opt.distributed)
            set_model_requires_grad(hiding_extractor, requires_grad=True, distributed=opt.distributed)
            set_model_requires_grad(discriminator, requires_grad=False, distributed=opt.distributed)

            with torch.no_grad():
                ori_id = arc_face(ori_image)
                denorm_ori_image = denormalizer(ori_image)
                ori_3d, _ = face_recon_net.compute_coeff(denorm_ori_image)
                ori_shape, ori_lm, ori_color = face_recon_net.compute_3d_shape(ori_3d)

                binary_hiding = float2bit(torch.randn([opt.batch_size, 512]).to(device))
            # anonymous
            rand_z = torch.randn([opt.batch_size, 512]).to(device)
            rand_id = map_2_id(rand_z)
            latent_control_anonymous = style_mlp(rand_id, torch.cat([ori_3d[:, 0:144], ori_3d[:, 224:227]], dim=1))
            anonymous_image = anonymous_net(ori_image, latent_control_anonymous, binary_hiding)
            
            de_norm_anonymous_image = denormalizer(anonymous_image)
            anonymous_3d, _ = face_recon_net.compute_coeff(de_norm_anonymous_image)
            anonymous_shape, anonymous_lm, anonymous_color = face_recon_net.compute_3d_shape(anonymous_3d)
            anonymous_id = arc_face(anonymous_image)
            
            # recon variable
            ori_image_recon = ori_image.detach().clone()
            ori_image_id = ori_id.detach().clone()
            anonymous_3d_recon = anonymous_3d.detach().clone() 
            anonymous_image_recon = anonymous_image.detach().clone()
            
            ori_shape_recon = ori_shape.detach().clone()
            ori_lm_recon = ori_lm.detach().clone()
            ori_color_recon = ori_color.detach().clone()
            
            fake_image_D = anonymous_image.detach().clone()

            anonymous_loss = 0.0
            # id dummy diversity
            id_dummy_diversity, _ = criterion_id_diversity(rand_id)
            anonymous_process_loss_dict['id_dummy_diversity'] = id_dummy_diversity
            anonymous_loss += id_dummy_diversity
            
            # id gen consist
            id_gen_consist, _ = criterion_id(rand_id, anonymous_id, ID_same_label)
            anonymous_process_loss_dict['id_gen_consist'] = id_gen_consist
            anonymous_loss += id_gen_consist
            
            # id 
            anonymous_id_loss, loss_item = criterion_id(ori_id, anonymous_id, ID_diff_label)
            anonymous_process_loss_dict.update(loss_item)
            anonymous_loss += anonymous_id_loss
            
            # geometry
            anonymous_geometry_loss, loss_item = criterion_geometry(ori_shape,
                                                                    anonymous_shape,
                                                                    ori_lm,
                                                                    anonymous_lm,
                                                                    ori_color,
                                                                    anonymous_color,
                                                                    ) 
            anonymous_process_loss_dict.update(loss_item)
            anonymous_loss += anonymous_geometry_loss
            
            # different from paper report, add pixel loss to stable training
            anonymous_pixel_loss, loss_item = criterion_recon(anonymous_image, ori_image)
            anonymous_process_loss_dict.update(loss_item)
            anonymous_loss += anonymous_pixel_loss
            
            # adv loss
            anonymous_discr_pred = discriminator(anonymous_image)
            anonymous_adv_loss, loss_item = criterion_generator(anonymous_discr_pred)
            anonymous_process_loss_dict.update(loss_item)
            anonymous_loss += anonymous_adv_loss

            # hiding
            noise = torch.randn_like(anonymous_image) * 0.01
            anonymous_image = anonymous_image + noise
            anonymous_extracting_info = hiding_extractor(anonymous_image)
            anonymous_hiding_loss, loss_item = criterion_info_hiding(anonymous_extracting_info, binary_hiding)
            anonymous_process_loss_dict.update(loss_item)
            anonymous_loss += anonymous_hiding_loss

            optimizer.zero_grad()
            anonymous_loss.backward()
            optimizer.step()

            recon_loss = 0.0
            binary_hiding = float2bit(torch.randn([opt.batch_size, 512]).to(device))
            latent_control_recon = style_mlp(ori_id, torch.cat([anonymous_3d_recon[:, 0:144], anonymous_3d_recon[:, 224:227]], dim=1))
            recon_image = anonymous_net(anonymous_image_recon, latent_control_recon, binary_hiding)
            if i % 2 == 0:
                fake_image_D = recon_image.detach().clone()
                
            de_norm_recon_image = denormalizer(recon_image)
            recon_3d, _ = face_recon_net.compute_coeff(de_norm_recon_image)
            recon_shape, recon_lm, recon_color = face_recon_net.compute_3d_shape(recon_3d)
            recon_id = arc_face(recon_image)
            
            recon_id_loss, loss_item = criterion_id(ori_image_id, recon_id, ID_same_label)
            reconstruct_process_loss_dict.update(loss_item)
            recon_loss += recon_id_loss
            
            recon_geometry_loss, loss_item = criterion_geometry(ori_shape_recon,
                                                                recon_shape,
                                                                ori_lm_recon,
                                                                recon_lm,
                                                                ori_color_recon,
                                                                recon_color,
                                                                ) 
            reconstruct_process_loss_dict.update(loss_item)
            recon_loss += recon_geometry_loss
            
            recon_pixel_loss, loss_item = criterion_recon(recon_image, ori_image_recon)
            reconstruct_process_loss_dict.update(loss_item)
            recon_loss += recon_pixel_loss
            
            recon_discr_pred = discriminator(recon_image)
            recon_adv_loss, loss_item = criterion_generator(recon_discr_pred)
            reconstruct_process_loss_dict.update(loss_item)
            recon_loss += recon_adv_loss
            
            recon_image = recon_image + noise
            recon_extracting_info = hiding_extractor(recon_image)
            recon_hiding_loss, loss_item = criterion_info_hiding(recon_extracting_info, binary_hiding)
            reconstruct_process_loss_dict.update(loss_item)
            recon_loss += recon_hiding_loss
            
            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()

            ######## model mode ########
            set_model_requires_grad(anonymous_net, requires_grad=False, distributed=opt.distributed)
            set_model_requires_grad(style_mlp, requires_grad=False, distributed=opt.distributed)
            set_model_requires_grad(hiding_extractor, requires_grad=False, distributed=opt.distributed)          
            set_model_requires_grad(discriminator, requires_grad=True, distributed=opt.distributed)

            # train Discriminator
            if opt.augment:
                real_image_D, _ = augment(real_image_D, ada_aug_p)
                fake_image_D, _ = augment(fake_image_D, ada_aug_p)

            real_discr_pred = discriminator(real_image_D)
            fake_discr_pred = discriminator(fake_image_D)

            d_regularize = iteration % opt.d_reg_every == 0
            disc_loss, loss_item = criterion_discriminator(real_image_D, real_discr_pred, fake_discr_pred, d_regularize)
            d_regularize = False

            if opt.augment and opt.augment_p == 0:
                ada_aug_p = ada_augment.tune(real_discr_pred)

            GAN_loss_dict.update(loss_item)
            optimizer_D.zero_grad()
            disc_loss.backward()
            optimizer_D.step()

            anonymous_process_loss_reduced = reduce_loss_dict(anonymous_process_loss_dict)
            reconstruct_process_loss_reduced = reduce_loss_dict(reconstruct_process_loss_dict)
            GAN_loss_reduced = reduce_loss_dict(GAN_loss_dict)
            
            id_dummy_div_loss += anonymous_process_loss_dict['id_dummy_diversity'].mean().item()
            id_gen_consist_loss += anonymous_process_loss_dict['id_gen_consist'].mean().item()
            anonymous_process_l1_loss += anonymous_process_loss_reduced['l1_loss'].mean().item()
            anonymous_process_lpips_loss += anonymous_process_loss_reduced['lpips_loss'].mean().item()
            anonymous_process_id_loss += anonymous_process_loss_reduced['id_loss'].mean().item()
            anonymous_process_shape_loss += anonymous_process_loss_reduced['shape_loss'].mean().item()
            anonymous_process_landmark_loss += anonymous_process_loss_reduced['landmark_loss'].mean().item()
            anonymous_process_color_loss += anonymous_process_loss_reduced['color_loss'].mean().item()
            anonymous_process_hiding_loss += anonymous_process_loss_reduced['hiding_loss'].mean().item() 
            anonymous_process_gen_loss += anonymous_process_loss_reduced['gen_loss'].mean().item()   

            reconstruct_process_l1_loss += reconstruct_process_loss_reduced['l1_loss'].mean().item()
            reconstruct_process_lpips_loss += reconstruct_process_loss_reduced['lpips_loss'].mean().item()
            reconstruct_process_id_loss += reconstruct_process_loss_reduced['id_loss'].mean().item()
            reconstruct_process_shape_loss += reconstruct_process_loss_reduced['shape_loss'].mean().item()
            reconstruct_process_landmark_loss += reconstruct_process_loss_reduced['landmark_loss'].mean().item()
            reconstruct_process_color_loss += reconstruct_process_loss_reduced['color_loss'].mean().item()
            reconstruct_process_hiding_loss += reconstruct_process_loss_reduced['hiding_loss'].mean().item()   
            reconstruct_process_gen_loss += reconstruct_process_loss_reduced['gen_loss'].mean().item()    

            disc_loss_epoch += GAN_loss_reduced['disc_loss'].mean().item()
            disc_real_loss += GAN_loss_reduced['real_loss'].mean().item()
            disc_fake_loss += GAN_loss_reduced['fake_loss'].mean().item()
            disc_gp_loss += GAN_loss_reduced['grad_penalty_loss'].mean().item()

            if get_rank() == 0:
                writer.add_scalar('Diversity/ID_dummy', anonymous_process_loss_dict['id_dummy_diversity'].mean(), iteration)
                
                writer.add_scalar('Anonymous/ID_consist', anonymous_process_loss_dict['id_gen_consist'].mean(), iteration)
                writer.add_scalar('Anonymous/L1', anonymous_process_loss_reduced['l1_loss'].mean(), iteration)
                writer.add_scalar('Anonymous/LPIPS', anonymous_process_loss_reduced['lpips_loss'].mean(), iteration)
                writer.add_scalar('Anonymous/ID', anonymous_process_loss_reduced['id_loss'].mean(), iteration)
                writer.add_scalar('Anonymous/Gen', anonymous_process_loss_reduced['gen_loss'].mean(), iteration)
                
                writer.add_scalar('Reconstruct/L1', reconstruct_process_loss_reduced['l1_loss'].mean(), iteration)
                writer.add_scalar('Reconstruct/LPIPS', reconstruct_process_loss_reduced['lpips_loss'].mean(), iteration)
                writer.add_scalar('Reconstruct/ID', reconstruct_process_loss_reduced['id_loss'].mean(), iteration)
                writer.add_scalar('Reconstruct/Gen', reconstruct_process_loss_reduced['gen_loss'].mean(), iteration)
                
                writer.add_scalar('D/disc', GAN_loss_reduced['disc_loss'].mean(), iteration)
                writer.add_scalar('D/real', GAN_loss_reduced['real_loss'].mean(), iteration)
                writer.add_scalar('D/fake', GAN_loss_reduced['fake_loss'].mean(), iteration)
                writer.add_scalar('Learning Rate',  optimizer.param_groups[0]['lr'], epoch)

            if (iteration+1) % 10 == 0:
                if get_rank() == 0:
                    # epoch loss
                    msg = '[iter: %06d/%d] | LR: %.05f | Diversity [dummy: %.03f consist: %.03f] | ' \
                        'Anonymous [ID: %.03f L1: %.03f LPIPS: %.03f shape: %.03f landmark: %.03f color: %.03f gen: %.03f hiding: %.03f] | ' \
                        'Reconstruct [ID: %.03f L1: %.03f LPIPS: %.03f shape: %.03f landmark: %.03f color: %.03f gen: %.03f hiding: %.03f] | ' \
                        'Adv Loss: [disc: %.03f real: %.03f fake: %.03f gp: %.03f]' % \
                        (iteration+1, opt.num_iter, cur_lr[0], id_dummy_div_loss / (i+1), id_gen_consist_loss / (i+1), 
                        anonymous_process_id_loss / (i+1), anonymous_process_l1_loss / (i+1), anonymous_process_lpips_loss / (i+1), 
                        anonymous_process_shape_loss / (i+1), anonymous_process_landmark_loss / (i+1), anonymous_process_color_loss / (i+1), 
                        anonymous_process_gen_loss / (i+1), anonymous_process_hiding_loss / (i+1), 
                        reconstruct_process_id_loss / (i+1), reconstruct_process_l1_loss / (i+1), reconstruct_process_lpips_loss / (i+1), 
                        reconstruct_process_shape_loss / (i+1), reconstruct_process_landmark_loss / (i+1), reconstruct_process_color_loss / (i+1), 
                        reconstruct_process_gen_loss / (i+1), reconstruct_process_hiding_loss / (i+1), 
                        disc_loss_epoch / (i+1), disc_real_loss / (i+1), disc_fake_loss / (i+1), disc_gp_loss / (i+1),
                        )                     
                    logger.info(msg)
            
            if iteration % 100 == 0:
                if get_rank() == 0:
                    img = torch.cat([ori_image, anonymous_image, recon_image], dim=0).detach().cpu()
                    img = img.clip(-1, 1)
                    save_image(img, '%s/image/train/%06d.jpg' % (work_path, iteration), nrow=opt.batch_size, normalize=True, scale_each=True)


            if opt.distributed:
                checkpoint = {'epoch': epoch,
                'iter': iteration,
                'anonymous_net': anonymous_net.module.state_dict(),
                'style_mlp': style_mlp.module.state_dict(),
                'map_2_id': map_2_id.module.state_dict(),
                'hiding_extractor': hiding_extractor.module.state_dict(),
                'discriminator': discriminator.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'scheduler': scheduler_warmup.state_dict(),
                }
            else:
                checkpoint = {'epoch': epoch,
                'iter': iteration,
                'anonymous_net': anonymous_net.state_dict(),
                'style_mlp': style_mlp.state_dict(),
                'map_2_id': map_2_id.state_dict(),
                'hiding_extractor': hiding_extractor.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer': optimizer.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'scheduler': scheduler_warmup.state_dict(),
                }
            
            if iteration % 10000 == 0:
                if get_rank() == 0:
                    torch.save(checkpoint, '%s/%d_checkpoint.pth' % (work_path, iteration))

                
        # test 
        test_images = []
        save_num = 0

        map_2_id.eval()
        style_mlp.eval()
        hiding_extractor.eval()
        anonymous_net.eval()

        for i, data in enumerate(test_loader, 0):
            with torch.no_grad():
                ori_image = data.cuda()
                ori_id = arc_face(ori_image)

                denorm_ori_image = denormalizer(ori_image)
                ori_3d, _ = face_recon_net.compute_coeff(denorm_ori_image)
                ori_shape, ori_lm, ori_color = face_recon_net.compute_3d_shape(ori_3d)
                binary_hiding = float2bit(ori_id)
                rand_z = torch.randn([opt.batch_size, 512]).to(device)

                rand_id = map_2_id(rand_z)
                latent_control = style_mlp(rand_id, torch.cat([ori_3d[:, 0:144], ori_3d[:, 224:227]], dim=1))
                anonymous_image = anonymous_net(ori_image, latent_control, binary_hiding)

                # test HidingExtractor
                extracting_info = hiding_extractor(anonymous_image)

                # hiding 
                extract_float = bit2float(torch.round(torch.sigmoid(extracting_info).cpu())).cuda()
                extract_float = torch.where(extract_float<-50, torch.full_like(extract_float, 0.0), extract_float)
                extract_float = torch.where(extract_float>50, torch.full_like(extract_float, 0.0), extract_float)

                # invert image
                denorm_anonymous_image = denormalizer(anonymous_image)
                anonymous_3d, _ = face_recon_net.compute_coeff(denorm_anonymous_image)

                latent_control = style_mlp(extract_float, torch.cat([anonymous_3d[:, 0:144], anonymous_3d[:, 224:227]], dim=1))
                invert_image = anonymous_net(anonymous_image, latent_control, binary_hiding)

            if get_rank() == 0:
                if save_num < 64:
                    test_images.append(torch.cat([ori_image[:opt.batch_size],
                                                  anonymous_image[:opt.batch_size], 
                                                  invert_image[:opt.batch_size]], 
                                                  dim=0).detach().cpu()
                                      )
                    save_num += opt.batch_size

        if get_rank() == 0:
            test_images = torch.cat(test_images, dim=0)
            test_images = test_images.clip(-1, 1)
            save_image(test_images, '%s/image/test/epoch_%05d.jpg' % (work_path, epoch), nrow=opt.batch_size, normalize=True, scale_each=True)
            save_num = 0

        torch.save(checkpoint, '%s/last_checkpoint.pth' % work_path)
        torch.cuda.empty_cache()
        
    torch.save(checkpoint, '%s/final_checkpoint.pth' % (work_path))  



if __name__ == '__main__':
    import datetime
    opt = options()

    if opt.distributed:
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
        
    start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    work_path = os.path.join(opt.work_path, start_time)

    if get_rank() == 0:
        os.makedirs(os.path.join(work_path, 'image', 'train'), exist_ok=True)
        os.makedirs(os.path.join(work_path, 'image', 'test'), exist_ok=True)

    logger = setup_logger(work_path)
    train(opt, logger, work_path)