import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
import networkx as nx
from utils import *


# this file is to preprocess the dataset , preparing for training the model

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s,
                    allowable_set))  # !!!map lambda function is hard to understand, but function of this code is knowned/aware






# from DeepDTA data
all_prots = []
datasets = ['kiba', 'davis']
for dataset in datasets:
    print('convert data from DeepDTA for ', dataset)
    fpath = 'data/' + dataset + '/'
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))  # train_fold is 5*19709 list
    train_fold = [ee for e in train_fold for ee in e]  # train_fold is 98545（5*19709）list
    valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))  # valid_fold is 19709-length list
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')
    drugs = []
    prots = []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)
    for t in proteins.keys():
        prots.append(proteins[t])
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)
    opts = ['train', 'test']
    for opt in opts:
        rows, cols = np.where(np.isnan(affinity) == False)
        if opt == 'train':
            rows, cols = rows[train_fold], cols[train_fold]
        elif opt == 'test':
            rows, cols = rows[valid_fold], cols[valid_fold]
        with open('data/' + dataset + '_' + opt + '.csv', 'w') as f:
            f.write('compound_iso_smiles,target_sequence,affinity\n')
            for pair_ind in range(len(rows)):
                ls = []
                ls += [drugs[rows[pair_ind]]]
                ls += [prots[cols[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]  # affinity 矩阵，行对应药物，列对应蛋白质吗？
                f.write(','.join(map(str, ls)) + '\n')  # 逗号分隔
    print('\ndataset:', dataset)
    print('train_fold:', len(train_fold))
    print('test_fold:', len(valid_fold))
    print('len(set(drugs)),len(set(prots)):', len(set(drugs)), len(set(prots)))
    all_prots += list(set(prots))


compound_iso_smiles = []
for dt_name in ['kiba', 'davis']:
    opts = ['train', 'test']
    for opt in opts:
        df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
        compound_iso_smiles += list(df['compound_iso_smiles'])
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)  # c_size, features, edge_index  all deliever to g
    smile_graph[smile] = g  # Using SMILES as the keys, the graph of SMILES as the content ,save the dictionary as smile_graph

datasets = ['davis', 'kiba']
# convert to PyTorch data format, preparing for training the model
for dataset in datasets:
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        df = pd.read_csv('data/' + dataset + '_train.csv')
        train_drugs, train_prots, train_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['affinity'])
        train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(train_Y)

        df = pd.read_csv('data/' + dataset + '_test.csv')
        test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['affinity'])
        test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(test_prots), np.asarray(test_Y)

        # make data PyTorch Geometric ready
        # call MyDataset class，deliver xd = train_drugs（drug SMILES, array）, xt = train_prots(protein string，array), y = train_Y（target，array）, smile_graph = smile_graph（dictionry for storing drugs' graph ）
        print('preparing ', dataset + '_train.pt in pytorch format!')
        train_data = MyDataset(root='data', dataset=dataset + '_train', xd=train_drugs, xt=train_prots, y=train_Y,
                                    smile_graph=smile_graph)
        print('preparing ', dataset + '_test.pt in pytorch format!')
        test_data = MyDataset(root='data', dataset=dataset + '_test', xd=test_drugs, xt=test_prots, y=test_Y,
                                   smile_graph=smile_graph)
        print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')
    else:
        print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')
