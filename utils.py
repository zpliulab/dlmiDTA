import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

#store some needed classes and functions.

# This class is to get the PyTorch-Geometric format processed data
class MyDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(MyDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])   #_use_new_zipfile_serialization=False
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y,smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD: list of SMILES,
    # XT: list of protein sequence
    # Y: list of labels (affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, y,smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # get the molecular graph representation  from smile_graph dictionary
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
            # get the ECFP of drug
            GCNData.fp = torch.LongTensor([morganfp(smiles)])
            # GET the K-mer feature of  protein sequence
            GCNData.kmer = torch.FloatTensor([getmarkov(target)])
            # get the label encoding of protein sequence for CNN
            GCNData.target = torch.LongTensor([myseq_cat(target)])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

# functions needed

def morganfp(input):
    strr = input
    m1 = Chem.MolFromSmiles(strr)  # 当分子为空值时，fingerprints全为0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1,2,nBits=1024).ToBitString()
    fp2 = list(fp1)
    fp3 = list(map(int, fp2))
    return fp3   #输出的是一维数值数组，1024长

def getmarkov(input):
    strr = input
    list=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

    data0 = np.zeros(20)
    n = len(strr)
    for m in range(n):
        for i in range(20):
            if strr[m] == list[i]:
                data0[i] += 1
                break
    data0 = data0.reshape(-1, )
    data0 = (data0 - np.mean(data0)) / np.std(data0)

    data1 = np.zeros((20, 20))
    n = len(strr)
    ct = 0
    for m in range(n - 1):
        ct = 0
        for i in range(20):
            if strr[m] == list[i]:
                for j in range(20):
                    if strr[m+1] == list[j]:
                        data1[i][j] += 1
                        ct += 1
                        break
            if ct==1:
                break
    data1 = data1.reshape(-1, )
    data1 = (data1 - np.mean(data1)) / np.std(data1)
    data2 = np.zeros((20, 20, 20))
    n = len(strr)
    for m in range(n - 2):
        ct = 0
        for i in range(20):
            if strr[m] == list[i]:
                for j in range(20):
                    if strr[m+1] == list[j]:
                        for k in range(20):
                            if strr[m+2] == list[k]:
                                data2[i][j][k] += 1
                                ct += 1
                                break
                    if ct == 1:
                        break
            if ct==1:
                break
    data2 = data2.reshape(-1, )
    data2 = (data2 - np.mean(data2)) / np.std(data2)
    data3 = np.concatenate((data0, data1, data2))
    return data3

def myseq_cat(prot):
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
    max_seq_len = 1000
    x = np.zeros(max_seq_len)
    #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据下标和数据，一般用在 for 循环当中。
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

# Metrics
def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)  #可以发现，argsort()是将X中的元素从小到大排序后，提取对应的索引index，然后输出到ind
    y = y[ind]   #由大到小排序 y是标签，f 是预测值
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci



