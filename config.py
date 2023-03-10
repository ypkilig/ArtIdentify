import argparse

def argparsers():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='se_resnext50', type=str)
    parser.add_argument('--savepath', default='./data/result', type=str)
    parser.add_argument('--readpath', default='../data-1/Art/train.csv', type=str)
    parser.add_argument('--loss', default='ce', type=str)
    parser.add_argument('--num_classes', default=49, type=int)
    parser.add_argument('--pool_type', default='avg', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--scheduler', default='cos', type=str)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--lr_step', default=25, type=int)
    parser.add_argument('--lr_gamma', default=0.1, type=float)
    parser.add_argument('--total_epoch', default=60, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--multi-gpus', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=3406, type=int)
    parser.add_argument('--pretrained', default=True, type=bool)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argparsers()
    print(args.seed, args.gpu)