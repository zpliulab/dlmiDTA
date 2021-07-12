import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from torch.autograd import Variable


# 85 行，得torch.LongTensor 格式？？ 加上.float()转换下就好了，根本原因在于MyDataset_1类中的数值类型转换错误了！！！！！
# 85 行，得torch.LongTensor 格式？？ 加上.float()转换下就好了，根本原因在于MyDataset_1类中的数值类型转换错误了！！！！！

# GCN based model
# graph /  fp  / squence / k-mer
class Gnn1101(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=16, num_features_xd=78, num_features_xt=25, output_dim=128, dropout=0.2):

        super(Gnn1101, self).__init__()
        self.n_output = n_output
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # SMILES graph branch
        self.conv1 = GCNConv(num_features_xd, num_features_xd)   #num_features_xd is the number of features, else is the size of hidden layer
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)


        # # protein sequence branch (1d conv)
        # self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        # self.conv_p1 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)
        # self.conv_p2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5)
        # self.conv_p3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        # self.conv_p4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        # # self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        # self.fc1_xt = nn.Linear(64 * 244, output_dim)

        # protein k-mer branch (fully connected layer)
        km_feature = 8420
        self.fc_km1 = nn.Linear(km_feature, 1024)
        self.fc_km2 = nn.Linear(1024, 512)
        self.fc_km3 = nn.Linear(512, output_dim)

        # drug fp branch (fully connected layer)
        fp_feature = 1024
        self.fc_fp1 = nn.Linear(fp_feature, 1024)
        self.fc_fp2 = nn.Linear(1024, 512)
        self.fc_fp3 = nn.Linear(512, output_dim)

        # combined layers
        self.fc1 = nn.Linear(3*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        # get graph input   X is the node feature matrix.,edge_index is the adjacency matrix
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # get protein input
        target = data.target
        km = data.kmer   #xk
        fp = data.fp     #xf

        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)       # global max pooling, num_features_xd*4 layers of n

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # # 1d conv layers
        # embedded_xt = self.embedding_xt(target)
        # embedded_xt = Variable(embedded_xt.permute(0, 2, 1))
        # out = F.relu(self.conv_p1(embedded_xt))
        # out1 = F.relu(self.conv_p2(out))
        # out2 = F.max_pool1d(out1, 2)
        # out3 = F.relu(self.conv_p3(out2))
        # out4 = F.relu(self.conv_p4(out3))
        # out5 = F.max_pool1d(out4, 2)
        # # out6 = out5[:,:,length/4]
        # # flatten
        # xt = out5.view(-1, 64 * 244)
        # xt = self.fc1_xt(xt)

        # protein kmer branch (fully connected layer)
        xk = self.relu(self.fc_km1(km))
        xk = self.relu(self.fc_km2(xk))
        xk = self.relu(self.fc_km3(xk))

        # drug fp branch (fully connected layer)
        xf = self.relu(self.fc_fp1(fp.float()))    #得torch.LongTensor 格式？？ 加上.float()转换下就好了，根本原因在于MyDataset_1类中的数值类型转换错误了！！！！！
        xf = self.relu(self.fc_fp2(xf))
        xf = self.relu(self.fc_fp3(xf))

        # concat
        xc = torch.cat((x,xf,xk), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

