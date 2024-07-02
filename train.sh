CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8888 train.py --distributed True \
 --celebahq_path YOUR_CELEBAHQ_PATH \
 --ffhq_path YOUR_FFHQ_PATH