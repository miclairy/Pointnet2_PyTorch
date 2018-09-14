from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from data.part_datasetss import PartDataset, normalize, transform
from pointnet import PointNetCls
from pointnet2 import PointNetCls2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data.meta_dataset import MetaDataset, pad

from models import Pointnet2ClsMSG as Pointnet
from models.pointnet2_msg_cls import model_fn_decorator

if __name__ == '__main__':
    #showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default = '',  help='model path')
    parser.add_argument('--num_points', type=int, default=2500, help='input batch size')


    opt = parser.parse_args()
    print (opt)
    path = '../meta_data'
    batch = 15
    print("Testing on: ", path)
    # test_dataset = MetaDataset(root = '../meta_data', train = False, transform=pad)
    test_dataset = PartDataset(root = path, classification = True, train = False, npoints = opt.num_points, transform = normalize)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch,
                                            shuffle=True, num_workers=4 )

    # classifier = PointNetCls(k = len(test_dataset.classes))
    classifier = Pointnet(3, input_channels=0, use_xyz=True)
    classifier.load_state_dict(torch.load(opt.model))
    # classifier.torch.load(opt.model)
    # classifier = torch.load(opt.model)
    classifier.cuda()

    classifier.eval()

    tot_correct = 0
    print(len(testdataloader))
    for i, data in enumerate(testdataloader, 0):
        points, target = data
        points, target = Variable(points), Variable(target[:, 0])
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        pred, _ = classifier(points)
        loss = F.nll_loss(pred, target)

        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        tot_correct += float(correct)
        
        print('i:{} loss: {} accuracy: {}'.format(i, loss.item(), float(correct) / batch))
        pred_choice = [int(c) for c in pred_choice]
        target = [int(c) for c in target]
        print(i,list(zip(pred_choice, target)))

    print("total accuracy: ", tot_correct / len(test_dataset))