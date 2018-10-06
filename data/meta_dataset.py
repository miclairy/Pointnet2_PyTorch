import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from sklearn import preprocessing
import pandas as pd

class MetaDataset(data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.data_files = {}
        self.categories = ['cast-off', 'impact', 'expirated']
        self.transform = transform

        for category in self.categories:
            self.data_files[category] = []
            dir_data = os.path.join(self.root, category)
            filenames = sorted(os.listdir(dir_data))
            
            if train:
                filenames = filenames[:int(len(filenames) * 0.7)]
            elif train == None:
                filenames = filenames[int(len(filenames) * 0.8):int(len(filenames) * 0.9)]
            else:
                filenames = filenames[int(len(filenames) * 0.9):]

            for filename in filenames:
                name = (os.path.splitext(os.path.basename(filename))[0])
                self.data_files[category].append((os.path.join(dir_data, name + '.csv')))

        self.datapath = []
        for category in self.categories:
            for filename in self.data_files[category]:
                self.datapath.append((category, filename))

        self.classes = dict(zip(sorted(self.categories), range(len(self.categories))))

    def __getitem__(self, index):
        filename = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]     
        data_set = pd.read_csv(filename[1])        
        
        data_set.insert(3, 'fake z', 0)

        headers = ['position x', 'position y', 'fake z', 'area px', 'area_mm', 
                            'width ellipse', 'height ellipse', 'angle', 'gamma', 
                             'solidity', 'circularity', 'intensity']
        one_hot_headers = ["('left', 'down')",  "('right', 'down')",  "('right', 'up')", "('left', 'up')",  '?']
        one_hot_encoding = pd.get_dummies(data_set['direction'])
        data_set = pd.concat([data_set[headers], one_hot_encoding], axis=1, sort=False)

        for i in range(len(one_hot_headers)):
            if one_hot_headers[i] not in data_set.columns.values.tolist():
                data_set.insert(i + len(headers), one_hot_headers[i], 0)

        if self.transform is not None:
            data_set = self.transform(data_set)

        data_set.fillna(-1, inplace=True)
        
        data_values = np.array(data_set[headers + one_hot_headers].values).astype(np.float32)
        data_set = torch.from_numpy(data_values)
        
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return data_set, cls

    def __len__(self):
        return len(self.datapath)


def normalize(data_set):
    x, y = data_set[['position x']].values.astype(float), data_set[['position y']].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    if x.shape[0] > 1:
        x_scaled = min_max_scaler.fit_transform(x)
        y_scaled = min_max_scaler.fit_transform(y)
        data_set['position x'] = x_scaled
        data_set['position y'] = y_scaled
    

    return data_set

def pad(data_set):
    data_set = normalize(data_set)
    headers = ['position x', 'position y', 'area px', 'area_mm', 
                            'width ellipse', 'height ellipse', 'angle', 'gamma', 
                             'solidity', 'circularity', 'intensity']
    one_hot_headers = ["('left', 'down')",  "('right', 'down')",  "('right', 'up')", "('left', 'up')",  '?']
    rows = data_set.shape[0]
    if (rows < 5000):
        empty = pd.DataFrame(np.ndarray((5000 - rows, 16), dtype=float).fill(-1), index=np.array(range(rows, 7500)), columns=headers+one_hot_headers)
        data_set = pd.concat([data_set, empty], sort=False)
    

    return data_set[:5000]


if __name__ == '__main__':
    d = MetaDataset(root = '../meta_data')
    print(len(d))
    ps, cls = d[1]
    print(ps, ps.type(), cls.size(), cls.type())
