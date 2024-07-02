
from torchvision.transforms import transforms
from .dataset import FaceHQDataset
from torch.utils.data import DataLoader
from torch.utils import data


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def data_transform():
    data_transform = transforms.Compose([transforms.Resize((256, 256)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    return data_transform


def setup_data(opt):
    train_dst = FaceHQDataset(celebahq_path=opt.celebahq_path,
                              ffhq_path=opt.ffhq_path,
                              transform=data_transform(),
                              mode='train',
                             )                     

    test_dst = FaceHQDataset(celebahq_path=opt.celebahq_path,
                             ffhq_path=opt.ffhq_path,
                             transform=data_transform(),
                             mode='test',
                            )
    
    train_loader = DataLoader(
        dataset=train_dst, 
        batch_size = opt.batch_size, 
        sampler=data_sampler(train_dst, shuffle=True, distributed=opt.distributed),
        drop_last=True,
        pin_memory=True,
        num_workers=8)
    
    test_loader = DataLoader(
        dataset=test_dst, 
        batch_size = opt.batch_size, 
        sampler=data_sampler(test_dst, shuffle=False, distributed=opt.distributed),
        drop_last=True,
        pin_memory=True,
        num_workers=8)

    return train_loader, test_loader