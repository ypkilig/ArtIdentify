from config import argparsers
from dataload import MyDataset, datafil
from ArtModel import BaseModel

import time
import os
import random

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.backends import cudnn
import numpy as np


def train():
    model.train()
    train_epoch_loss, correct, total = 0, 0., 0.
    st = time.time()
    for data, labels in trainloader:
        data, labels = data.to(device), labels.to(device)
        output = model(data)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_epoch_loss += loss.item()*data.size(0)
        total += data.size(0)
        _, pred = torch.max(output, 1)
        correct += pred.eq(labels).sum().item()
    acc = correct / total
    loss = train_epoch_loss / total

    print(f'loss:{loss:.4f} acc@1:{acc:.4f} time:{time.time() - st:.2f}s', end=' --> ')
    
    with open(os.path.join(savepath, 'log.txt'), 'a+')as f:
        f.write('loss:{:.4f}, acc:{:.4f} ->'.format(loss, acc))
    return {'loss': loss, 'acc': acc}

def test(epoch):
    model.eval()

    test_epoch_loss, correct, total = 0, 0., 0. 

    with torch.no_grad():
        for data, labels in valloader:
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            loss = criterion(output, labels)
            test_epoch_loss += loss.item() * data.size(0)
            total += data.size(0)
            _, pred = torch.max(output, 1)
            correct += pred.eq(labels).sum().item()

        acc = correct / total
        loss = test_epoch_loss / total

        print(f'test loss:{loss:.4f} acc@1:{acc:.4f}', end=' ')
    global best_acc, best_epoch
    state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch
            }
    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        torch.save(state, os.path.join(savepath, 'best.pth'))
        print('*')
    else:
        print()

    torch.save(state, os.path.join(savepath, 'last.pth'))


    with open(os.path.join(savepath, 'log.txt'), 'a+')as f:
        f.write('epoch:{}, loss:{:.4f}, acc:{:.4f}\n'.format(epoch, loss, acc))

    return {'loss': loss, 'acc': acc}


def plot(d, mode='train', best_acc_=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.suptitle('%s_curve' % mode)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    epochs = len(d['acc'])

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(epochs), d['loss'], label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(epochs), d['acc'], label='acc')
    if best_acc_ is not None:
        plt.scatter(best_acc_[0], best_acc_[1], c='r')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend(loc='upper left')

    plt.savefig(os.path.join(savepath, '%s.jpg' % mode), bbox_inches='tight')
    plt.close()
        




if __name__ == "__main__":

    args = argparsers()
    # datafil(args.readpath)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = False
        print('use seed: %s' %(args.seed))
    
    if torch.cuda.is_available():
        cudnn.benchmark = True
    
    criterion = nn.CrossEntropyLoss()

    trainloader = DataLoader(dataset=MyDataset("train"), batch_size=args.batch_size, 
                             shuffle=True, 
                             num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    valloader = DataLoader(dataset=MyDataset("val"), batch_size=32, 
                           shuffle=False, num_workers=args.num_workers, 
                           pin_memory=True)
    
    model = BaseModel(model_name=args.model_name ,num_classes=args.num_classes, pretrained=args.pretrained, pool_type=args.pool_type)

    if args.resume:
        state = torch.load(args.resume)
        print('best_epoch:{}, best_acc:{}'.format(state['epoch'], state['acc']))
        model.load_state_dict(state['net'])

    if torch.cuda.device_count() > 1 and args.multi_gpus:
        print('use multi-gpus...')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.distributed.init_process_group(backend="nccl", init_method='tcp://localhost:23456', rank=0, world_size=2)
        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(model)
    else:
        device = ('cuda:%d'%args.gpu if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    print('device:', device)

    # optim
    optimizer = torch.optim.SGD(
            [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr}],
            weight_decay=args.weight_decay, momentum=args.momentum)

    print('init_lr={}, weight_decay={}, momentum={}'.format(args.lr, args.weight_decay, args.momentum))

    assert args.scheduler in ["step", "multi",  "cos"]
    if args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma, last_epoch=-1)
    elif args.scheduler == 'multi':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225], gamma=args.lr_gamma, last_epoch=-1)
    elif args.scheduler == 'cos':
      warm_up_step = 10
      lambda_ = lambda epoch: (epoch + 1) / warm_up_step if epoch < warm_up_step else 0.5 * (
                    np.cos((epoch - warm_up_step) / (args.total_epoch - warm_up_step) * np.pi) + 1)
      scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_)
    
    # savepath
    savepath = os.path.join(args.savepath, args.model_name+args.pool_type+'_'+args.scheduler)

    print('savepath:', savepath)

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    with open(os.path.join(savepath, 'setting.txt'), 'w')as f:
        for k, v in vars(args).items():
            f.write('{}:{}\n'.format(k, v))

    f = open(os.path.join(savepath, 'log.txt'), 'w')
    f.close()

    start = time.time()

    train_info = {'loss': [], 'acc': []}
    test_info = {'loss': [], 'acc': []}
    best_acc = 0.
    best_epoch = 0
    for epoch in range(args.total_epoch):
        print('epoch[{:>3}/{:>3}]'.format(epoch, args.total_epoch), end=' ')
        d_train = train()
        scheduler.step()
        d_test = test(epoch)

        for k in train_info.keys():
            train_info[k].append(d_train[k])
            test_info[k].append(d_test[k])

        plot(train_info, mode='train')
        plot(test_info, mode='test', best_acc_=[best_epoch, best_acc])

    end = time.time()
    print('total time:{}m{:.2f}s'.format((end - start) // 60, (end - start) % 60))
    print('best_epoch:', best_epoch)
    print('best_acc:', best_acc)
    with open(os.path.join(savepath, 'log.txt'), 'a+')as f:
        f.write('# best_acc:{:.4f}, best_epoch:{}'.format(best_acc, best_epoch))

