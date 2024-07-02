import os
from torch.utils.data import Dataset
from PIL import Image


class FaceHQDataset(Dataset):
    def __init__(self, celebahq_path, ffhq_path, transform, mode='train'):
        super(FaceHQDataset, self).__init__()
        self.image_path_list = []
        celebahq_list = os.listdir(celebahq_path)
        ffhq_list = os.listdir(ffhq_path)
        ffhq_list.sort()
        if mode == 'train':
            for file in ffhq_list:
                self.image_path_list.append(os.path.join(ffhq_path, file))
        elif mode == 'test':
            for file in celebahq_list:
                self.image_path_list.append(os.path.join(celebahq_path, file))
        else:
            raise NotImplementedError
        
        self.transform = transform

    def __getitem__(self, index):
        try:
            image_name = self.image_path_list[index]
            img = Image.open(image_name)
            if self.transform is not None:
                img = self.transform(img)
        except:
            print(self.image_path_list[index])
        return img

    def __len__(self):
        return len(self.image_path_list)


class ImageDataset(Dataset):
    def __init__(self, root, transform):
        super(ImageDataset, self).__init__()
        self.root = root
        self.image_list = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.image_list[index]
        img = Image.open(os.path.join(self.root, image_name))
        if self.transform is not None:
            img = self.transform(img)
        return img, image_name

    def __len__(self):
        return len(self.image_list)