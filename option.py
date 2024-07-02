import argparse


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true',default=False)
    parser.add_argument('--seed', type=int, default=2024, help='random_seed')
    parser.add_argument('--resume_weights',type=str, default='', help='path of resume weights')

    # system setting
    parser.add_argument("--work_path", type=str, default='./run', help="result path")  
    parser.add_argument("--distributed", type=bool, default=False, help="whether use DDP training")
    parser.add_argument("--local-rank", type=int, default=-1, help="local rank for distributed training")
    parser.add_argument("--device", type=str, default='cuda', help="CUDA devices")

    # training setting
    parser.add_argument("--celebahq_path", type=str, default="", help="testing data path")
    parser.add_argument("--ffhq_path", type=str, default="", help="training data path")
    parser.add_argument('--image_size', default=256, type=int, help='image resolution') 
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate') 
    parser.add_argument('--batch_size', default=8, type=int, help='batch size on each gpu')
    parser.add_argument('--num_iter', default=100000, type=int, help='number of iterations')
    parser.add_argument("--d_reg_every", type=int, default=16, help="interval of the applying r1 regularization")
    parser.add_argument("--augment", action="store_true", help="apply non leaking augmentation")
    parser.add_argument("--augment_p", type=float, default=0, help="probability of applying augmentation. 0 = use adaptive augmentation")
    parser.add_argument("--ada_target", type=float, default=0.6, help="target augmentation probability for adaptive augmentation")
    parser.add_argument("--ada_length",type=int, default=500 * 1000, help="target duraing to reach augmentation probability for adaptive augmentation")
    parser.add_argument("--ada_every",type=int, default=256, help="probability update interval of the adaptive augmentation")

    # hyper parameter
    parser.add_argument('--latent_dim', default=659, type=int, help='latent dim, 512+144+3') 
    parser.add_argument('--style_dim', default=512, type=int, help='style code dim') 
    parser.add_argument('--n_mlp', default=4, type=int, help='number of mlp layers') 
    parser.add_argument('--l1_coef', default=1, type=float, help='L1 loss coefficient') 
    parser.add_argument('--lpips_coef', default=1, type=float, help='LPIPS loss coefficient') 
    parser.add_argument('--id_coef', default=1, type=float, help='ID loss coefficient') 
    parser.add_argument('--vert_coef', default=1, type=float, help='Vert loss coefficient') 
    parser.add_argument('--tex_coef', default=1, type=float, help='Textual loss coefficient') 
    parser.add_argument('--col_coef', default=1, type=float, help='Color loss coefficient') 
    parser.add_argument('--lm_coef', default=0.01, type=float, help='Landmark loss coefficient') 
    parser.add_argument('--hiding_coef', default=10, type=float, help='Hiding loss coefficient') 
    parser.add_argument('--diversity_coef', default=1, type=float, help='ID diversity loss coefficient') 
    parser.add_argument('--gp_coef', default=5, type=float, help='Gradient penalty loss coefficient') 

    # net structure and parameters
    parser.add_argument('--isTrain', type=bool, default=False, help='zero initialize the last fc')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='network structure')
    parser.add_argument('--init_path', type=str, default='weights/epoch_20.pth')
    parser.add_argument('--use_last_fc', type=bool, default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='model/d3dfr/BFM')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')
    parser.add_argument('--checkpoints_dir', type=str, default='./weights')
    parser.add_argument('--name', type=str, default='', help='bfm model')

    # renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)
    parser.add_argument('--use_opengl', type=bool, default=True, help='use opengl context or not')
    args = parser.parse_args()
    return args