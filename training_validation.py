import sys, os
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

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * 512,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

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


datasets = [['davis','kiba'][int(sys.argv[1])]] 
modeling = [Gnn0111, Gnn1011, Gnn1101, Gnn1110, Gnn1001, Gnn0110, Gnn0101, Gnn1010, Gnn1111][int(sys.argv[2])]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = ["cuda:0","cuda:1"][int(sys.argv[3])]
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005  #LR = 0.0005   200, 0.001; 300, 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 200

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset )
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset+'_train')
        test_data = TestbedDataset(root='data', dataset=dataset+'_test')
        
        # split data into train data and valid data
        train_size = int(0.8 * len(train_data))
        valid_size = len(train_data) - train_size
        train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])        
        
        
        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        test_loader  = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_mse = 1000
        best_test_mse = 1000
        best_test_ci = 0
        best_epoch = -1
        model_file_name = 'Vali_model_' + model_st + '_' + dataset +  '.pkl'
        result_file_name = 'Vali_result_' + model_st + '_' + dataset +  '.csv'
        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch+1)
            print('predicting for valid data...')
            G,P = predicting(model, device, valid_loader)
            val = mse(G,P)
            if val<best_mse:
                best_mse = val
                best_epoch = epoch+1
                torch.save(model.state_dict(), model_file_name)
                print('predicting for test data')
                G,P = predicting(model, device, test_loader)
                ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]
                with open(result_file_name,'w') as f:
                    f.write(','.join(map(str,ret)))
                best_test_mse = ret[1]
                best_test_ci = ret[-1]
                print(best_mse, 'rmse improved at epoch ', best_epoch, '; In testset, best_test_mse,best_test_ci:', best_test_mse,best_test_ci,model_st,dataset)
            else:
                print(ret[1],'No improvement since epoch ', best_epoch, '; best_test_mse,best_test_ci:', best_test_mse,best_test_ci,model_st,dataset)

