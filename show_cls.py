import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import os
import tensorboard_logger as tb_log

from models import Pointnet2ClsMSG as Pointnet
from models.pointnet2_msg_cls import model_fn_decorator
from data.part_datasets import PartDataset, normalize, transform
from data.meta_dataset import MetaDataset, pad
from data import ModelNet40Cls
import utils.pytorch_utils as pt_utils
import data.data_utils as d_utils
import argparse
import tqdm

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for cls training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-batch_size", type=int, default=10, help="Batch size")
    parser.add_argument(
        "-num_points",
        type=int,
        default=1024,
        help="Number of points to train with"
    )
    parser.add_argument(
        "-weight_decay",
        type=float,
        default=1e-5,
        help="L2 regularization coeff"
    )
    parser.add_argument(
        "-lr", type=float, default=1e-2, help="Initial learning rate"
    )
    parser.add_argument(
        "-lr_decay", type=float, default=0.7, help="Learning rate decay gamma"
    )
    parser.add_argument(
        "-decay_step", type=int, default=20, help="Learning rate decay step"
    )
    parser.add_argument(
        "-bn_momentum",
        type=float,
        default=0.5,
        help="Initial batch norm momentum"
    )
    parser.add_argument(
        "-bnm_decay",
        type=float,
        default=0.5,
        help="Batch norm momentum decay gamma"
    )
    parser.add_argument(
        "-checkpoint", type=str, default=None, help="Checkpoint to start from"
    )
    parser.add_argument(
        "-epochs", type=int, default=200, help="Number of epochs to train for"
    )
    parser.add_argument(
        "-run_name",
        type=str,
        default="cls_run_1",
        help="Name for run in tensorboard_logger"
    )

    return parser.parse_args()


lr_clip = 1e-5
bnm_clip = 1e-2

if __name__ == "__main__":
    args = parse_args()

    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    transforms = transforms.Compose([
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudScale(),
        d_utils.PointcloudRotate(),
        d_utils.PointcloudRotatePerturbation(),
        d_utils.PointcloudTranslate(),
        d_utils.PointcloudJitter(),
        d_utils.PointcloudRandomInputDropout()
    ])

    test_set = PartDataset(root = '/media/cba62/Elements/high_thres', classification = True, train=False, npoints = args.num_points, transform = None)
    # test_set = MetaDataset(root = '/media/cba62/Elements/old-Meta_Data', train = None, transform=pad)

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print('LOADED')

    tb_log.configure('runs/{}'.format(args.run_name))

    model = Pointnet(3, input_channels=0, use_xyz=True, base_features=16)
    model.cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_lbmd = lambda e: max(args.lr_decay**(e // args.decay_step), lr_clip / args.lr)
    bn_lbmd = lambda e: max(args.bn_momentum * args.bnm_decay**(e // args.decay_step), bnm_clip)
    
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())


    if args.checkpoint is not None:
       
        filename = "{}.pth.tar".format(args.checkpoint.split('.')[0]) 
        if os.path.isfile(filename):
            print("==> Testing checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            epoch = checkpoint['epoch']
            it = checkpoint.get('it', 0.0)
            best_prec = checkpoint['best_prec']
            if checkpoint['model_state'] is not None:
                model.load_state_dict(checkpoint['model_state'])
            if optimizer is not None and checkpoint['optimizer_state'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state'])

            model.eval()

            eval_dict = {}
            total_loss = 0.0
            count = 0.0
            correct = 0

            for i, data in tqdm.tqdm(enumerate(test_loader, 0), total=len(test_loader),
                                    leave=False, desc='val'):
                optimizer.zero_grad()

                _, loss, eval_res = model_fn(model, data, eval=True)

                total_loss += loss.data.item()
                count += eval_res['size']
                correct += eval_res['correct'].item()
                for k, v in eval_res.items():
                    if v is not None:
                        eval_dict[k] = eval_dict.get(k, []) + [v]

            print("test loss {} accuracy {}".format(total_loss / count, correct / count))




    

    