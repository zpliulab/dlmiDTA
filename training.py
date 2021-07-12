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
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


dataset = 'kiba'   # or davis
modeling = [Gnn0101, Gnn1001, Gnn0110, Gnn1010, Gnn0111, Gnn1011, Gnn1101, Gnn1110, Gnn1111, Gnn1001, Gnn0110]  # Gnn1001, Gnn0110,
cuda_name = "cuda:0"
print('cuda_name:', cuda_name)
LOG_INTERVAL = 20
NUM_EPOCHS = 200
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main loop: iterate over different models
for model in modeling:
    model_st = model.__name__
    print('\nrunning on ', model_st + '_' + dataset)
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = MyDataset(root='data', dataset=dataset + '_train')
        test_data = MyDataset(root='data', dataset=dataset + '_test')

        # loding the dataset
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = model().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_mse = 1000
        best_ci = 0
        best_epoch = -1
        result_file_name = 'result_' + dataset + '_' + model_st + '.csv'
        ret = np.zeros(5)
        para_file_name = 'Para'+ dataset + '_' + model_st + '.pkl'
        for epoch in range(NUM_EPOCHS + 5):
            train(model, device, train_loader, optimizer, epoch + 1)
            if epoch == NUM_EPOCHS-1:
                torch.save(model.state_dict(), para_file_name)
            if epoch >= NUM_EPOCHS:
                G, P = predicting(model, device, test_loader)
                ret += np.array([rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)])
                print('At epoch: ', epoch + 1, 'MSE', mse(G, P), '; CI', ci(G, P))
            if epoch == NUM_EPOCHS + 4:
                with open(result_file_name, 'w') as f:
                    f.write(','.join(map(str, ret / 5)))
                print('The avarage metrics:', 'MSE', ret[1] / 5, '; CI', ret[-1] / 5)
