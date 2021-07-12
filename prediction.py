import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from utils import *
from models.GNN0101 import Gnn0101
from models.GNN0110 import Gnn0110
from models.GNN1001 import Gnn1001
from models.GNN1111 import Gnn1111
from models.GNN1010 import Gnn1010
from models.GNN1110 import Gnn1110
from models.GNN1101 import Gnn1101
from models.GNN1011 import Gnn1011
from models.GNN0111 import Gnn0111

from utils import *

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


datasets = ['davis','kiba']
modelings = [Gnn0101, Gnn1001, Gnn0110, Gnn1010, Gnn0111, Gnn1011, Gnn1101, Gnn1110, Gnn1111, Gnn1001, Gnn0110]
cuda_name = "cuda:0"
print('cuda_name:', cuda_name)
TEST_BATCH_SIZE = 512

result = []
for dataset in datasets:
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if (not os.path.isfile(processed_data_file_test)):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        test_data = MyDataset(root='data', dataset=dataset+'_test')
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        for modeling in modelings:
            model_st = modeling.__name__
            print('\npredicting for ', dataset, ' using ', model_st)
            # training the model
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            model = modeling().to(device)
            model_file_name = 'pretrained/model_' + model_st + '_' + dataset +  '.model'
            if os.path.isfile(model_file_name):
                model.load_state_dict(torch.load(model_file_name))
                G,P = predicting(model, device, test_loader)
                ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]
                ret =[dataset, model_st] +  [round(e,3) for e in ret]
                result += [ ret ]
                print('dataset,model,rmse,mse,pearson,spearman,ci')
                print(ret)
            else:
                print('model is not available!')
with open('result.csv','w') as f:
    f.write('dataset,model,rmse,mse,pearson,spearman,ci\n')
    for ret in result:
        f.write(','.join(map(str,ret)) + '\n')

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


datasets = ['davis']
modelings = [Gnn1001]
cuda_name = "cuda:0"
print('cuda_name:', cuda_name)

TEST_BATCH_SIZE = 512
def is_array(var):
    return isinstance(var, tuple)
result = []
for dataset in datasets:
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if (not os.path.isfile(processed_data_file_test)):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        test_data = MyDataset(root='data', dataset=dataset+'_test')
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        for modeling in modelings:
            model_st = modeling.__name__
            print('\npredicting for ', dataset, ' using ', model_st)
            # training the model
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            model = modeling().to(device)
            para_file_name = 'Para' + dataset + '_' + model_st + '.pkl'
            result_file_name = 'Predict_result_' + dataset + '_' + model_st + '.csv'
            if os.path.isfile(model_file_name):
                model.load_state_dict(torch.load(para_file_name))
                G,P = predicting(model, device, test_loader)
                ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P), ci(G,P)]   #计算CI值的函数特别慢!!!!!!!
                print('dataset,model,rmse,mse,pearson,spearman,ci')
                print(dataset, model_st, ret)
            else:
                print('model is not available!')
with open('result.csv','w') as f:
    f.write('dataset,model,rmse,mse,pearson,spearman,ci\n')
    for ret in result:
        f.write(','.join(map(str,ret)) + '\n')