import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import progressbar
import sys
import torchvision.transforms as transforms
import argparse
import json
import random
import math


class PartDataset(data.Dataset):
    def __init__(self, root, npoints = None, classification = False, class_choice = None, train = True, transform=None):
        self.npoints = npoints
        self.root = root
        self.categories = ['cast-off', 'impact', 'expirated']

        self.classification = classification
        self.transform = transform

        self.meta = {}
        for category in self.categories:
            self.meta[category] = []
            dir_point = os.path.join(self.root, category)
            fns = sorted(os.listdir(dir_point))
            if train:
                fns = fns[:int(len(fns) * 0.7)]
            elif train == None:
                fns = fns[int(len(fns) * 0.8):int(len(fns) * 0.9)]
            else:
                fns = fns[int(len(fns) * 0.9):]

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[category].append((os.path.join(dir_point, token + '.pts')))

        self.datapath = []
        for category in self.categories:
            for fn in self.meta[category]:
                self.datapath.append((category, fn))

        self.classes = dict(zip(sorted(self.categories), range(len(self.categories))))
        self.num_seg_classes = 0


    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)

        point_set = torch.from_numpy(point_set)

        if self.npoints is not None:
            inds = torch.LongTensor(self.npoints).random_(0, point_set.size(0))
            point_set = point_set[inds]

        if self.transform is not None:
            point_set = self.transform(point_set)

        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        if self.classification:
            return point_set, cls

    def __len__(self):
        return len(self.datapath)



def normalize(point_set):
    x, y = point_set[:, 0], point_set[:, 1]

    x.add_(-x.mean().item()).div_(1000)
    y.add_(-y.mean().item()).div_(1000)

    return point_set


def random_log(l, u):
    return math.exp(random.uniform(math.log(l), math.log(u)))



def scaling(sx, sy):
    return torch.Tensor ([
      [sx, 0, 0],
      [0, sy, 0],
      [0, 0, 1]])

def rotation(a):
    sa = math.sin(a)
    ca = math.cos(a)

    return torch.Tensor ([
      [ca, -sa, 0],
      [sa,  ca, 0],
      [0,   0, 1]])

def translation(tx, ty):
    return torch.Tensor ([
      [1, 0, tx],
      [0, 1, ty],
      [0, 0, 1]])




def transform(translation_range=(-0.1, 0.1), scale_range=(0.8, 1.25), rotation_range=(-25, 25)):
    def f(point_set):
        point_set = normalize(point_set)

        x, y = point_set[:, 0], point_set[:, 1]
        cx, cy = x.mean(), y.mean()

        toCentre = translation(-cx, -cy)
        fromCentre = translation(cx, cy)

        scale = random.uniform(*scale_range)
        tx, ty = random.uniform(*translation_range), random.uniform(*translation_range)

        flip = 1 if random.uniform(0, 1) > 0.5 else -1

        t = translation(tx, ty)
        s = scaling(scale * flip, scale)

        r = rotation(random.uniform(*rotation_range))
        m = fromCentre.mm(r).mm(s).mm(t).mm(toCentre)

        point_set[:, 2].fill_(1)
        transformed = m.mm(point_set.t()).t()

        transformed[:, 2].fill_(0)

        # print(transformed.mean(0), point_set.mean(0), cx, cy)

        return transformed


    return f


if __name__ == '__main__':
    print('test')
    d = PartDataset(root = '../PointNet_Data',  classification = True)
    print(len(d))
    ps, cls = d[1]
    print(ps.size(), ps.type(), cls.size(), cls.type())

    # d = PartDataset(root = '../shapenetcore_partanno_segmentation_benchmark_v0', classification = True)
    # print(len(d))
    # ps, cls = d[0]
    # print(ps.size(), ps.type(), cls.size(),cls.type())
