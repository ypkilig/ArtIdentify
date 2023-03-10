import torch
from ArtModel import BaseModel
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
import numpy as np
import argparse


def get_setting(path):
    args = {}
    with open(os.path.join(path, 'setting.txt'), 'r')as f:
        for i in f.readlines():
            k, v = i.strip().split(':')
            args[k] = v
    return args


def load_pretrained_model(path, model, mode='best'):
    print('load pretrained model...')
    state = torch.load(os.path.join(path, '%s.pth' % mode))
    print('best_epoch:{}, best_acc:{}'.format(state['epoch'], state['acc']))
    model.load_state_dict(state['net'])


if __name__ == '__main__':
    mode = 'best'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--savepath', type=str, help="position")
    parser.add_argument('--last', action='store_true')
    args = parser.parse_args()
    path = args.savepath
    if args.last:
        mode = 'last'
    
    args = get_setting(path)

    model = BaseModel(model_name=args['model_name'], num_classes=int(args['num_classes']), \
        pretrained=int(args['pretrained']), pool_type=args['pool_type'], down=int(args['down']))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    load_pretrained_model(path, model, mode=mode)

    size = 512
    trans = transforms.Compose([
        transforms.Resize((int(size / 0.875), int(size / 0.875))),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    submit = {'uuid': [], 'label': []}
    TTA_times = 7

    model.eval()
    with torch.no_grad():
        for i in range(0, 800):
            img_path = './data/Art/data/test/%d.jpg' % i
            raw_img = Image.open(img_path).convert('RGB')
            results = np.zeros(49)

            for j in range(TTA_times):
                img = trans(raw_img)
                img = img.unsqueeze(0).to(device)
                out = model(img)
                out = torch.softmax(out, dim=1)
                _, pred = torch.max(out.cpu(), dim=1)

                results[pred] += 1
            pred = np.argmax(results)
            print(i, ',', pred)
            submit['uuid'].append(i)
            submit['label'].append(pred)

    df = pd.DataFrame(submit)
    df.to_csv(os.path.join(path, 'result.csv'), encoding='utf-8', index=False, header=False)