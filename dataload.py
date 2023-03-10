from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def datafil(path):
    df = pd.read_csv(path).values
    classFil = {}

    for idx, label in df:
        if label not in classFil:
            classFil[label] = []
        classFil[label].append(idx)

    big_x = []
    big_y = []
    small_x = []
    small_y = []

    for k, v in classFil.items():
        if len(v) < 30:
            small_x.extend(v)
            small_y.extend(np.ones(len(v), dtype=np.int16) * k)
        else:
            big_x.extend(v)
            big_y.extend(np.ones(len(v), dtype=np.int16) * k)

    train_x, test_x, train_y, test_y = train_test_split(big_x, big_y, random_state=3407, test_size=0.2)
    train_x.extend(small_x)
    train_y.extend(small_y)

    with open('./data/train.txt', 'w')as f:
        for fn, label in zip(train_x, train_y):
            f.write('../data-1/Art/train/{}.jpg,{}\n'.format(fn, label))

    with open('./data/val.txt', 'w')as f:
        for fn, label in zip(test_x, test_y):
            f.write('../data-1/Art/train/{}.jpg,{}\n'.format(fn, label))


class MyDataset(Dataset):
    """数据读取 配合 dataload
    """
    def __init__(self, mode):
        assert mode in ['train', 'val']
        txt = './data/%s.txt' % mode

        size = 512
        trans = {
                'train':
                    transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.ColorJitter(brightness=0.126, saturation=0.5),
                        transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), fillcolor=0, scale=(0.8, 1.2), shear=None),
                        transforms.Resize((int(size / 0.875), int(size / 0.875))),
                        transforms.RandomCrop((size, size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
                    ]),
                'val':
                    transforms.Compose([
                        transforms.Resize((int(size / 0.875), int(size / 0.875))),
                        transforms.CenterCrop((size, size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                }

        fpath = []
        labels = []
        with open(txt, 'r')as f:
            for i in f.readlines():
                fp, label = i.strip().split(',')
                fpath.append(fp)
                labels.append(int(label))

        self.fpath = fpath
        self.labels = labels
        self.mode = mode
        self.trans = trans[mode]
        
    def __getitem__(self, index):
        fp = self.fpath[index]
        label = self.labels[index]
        img = Image.open(fp).convert('RGB')
        if self.trans is not None:
            img = self.trans(img)

        return img, label

    def __len__(self):
        return len(self.labels)


