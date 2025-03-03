
import os
import csv
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

from models.GNNs import GCN, GraphSAGE, MLP, GAT, GIN
from models.tools import confirm_dir, load_conf
from dataset.utils import setup_seed
from dataset.loader import Dataset

class Predictor:
    def __init__(self, conf, data, method,  device='cuda:0', seed=0):
        super(Predictor, self).__init__()
        self.conf = conf
        self.device = torch.device(device)
        self.seed = seed
        self.method = method
        self.general_init(data)
        self.method_init(conf)

    def general_init(self, data):
        '''
        This conducts necessary operations for an experiment, including the setting specified split,
        variables to record statistics.
        '''
        self.loss_fn = F.binary_cross_entropy_with_logits if data.n_classes == 1 else F.cross_entropy
        self.edge_index = data.adj_coo.indices()
        self.adj = data.adj if self.conf.dataset['sparse'] else data.adj.to_dense()
        self.feats = data.feats
        self.n_nodes = data.n_nodes
        self.n_classes = data.n_classes
        self.clean_label = data.labels
        self.noisy_labels = data.noisy_labels
        self.weights = None

        self.loss_trajectories = np.zeros([self.n_nodes, self.conf.training['n_epochs']])  

    def method_init(self, conf):

        if self.method == 'gcn':
            self.model = GCN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'], out_channels=conf.model['n_classes'],
                            n_layers=conf.model['n_layer'], dropout=conf.model['dropout'],
                            norm_info=conf.model['norm_info'],
                            act=conf.model['act'], input_layer=conf.model['input_layer'],
                            output_layer=conf.model['output_layer']).to(self.device)
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                        weight_decay=self.conf.training['weight_decay'])
        elif self.method == 'sage':
            self.model = GraphSAGE(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'], out_channels=conf.model['n_classes'],
                            n_layers=conf.model['n_layer'], 
                            dropout=conf.model['dropout']).to(self.device)
            
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                        weight_decay=self.conf.training['weight_decay'])
        elif self.method == 'mlp':
            self.model = MLP(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'],
                         out_channels=conf.model['n_classes'],
                         n_layers=conf.model['n_layer'], dropout=conf.model['dropout']).to(self.device)
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                        weight_decay=self.conf.training['weight_decay'])
        elif self.method == 'gat':
            self.model = GAT(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'], out_channels=conf.model['n_classes'],
                         n_layers=conf.model['n_layer'], 
                         dropout=conf.model['dropout']).to(self.device)
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                        weight_decay=self.conf.training['weight_decay'])

        elif self.method == 'gin':
            self.model = GIN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'], out_channels=conf.model['n_classes'],
                         n_layers=conf.model['n_layer'], mlp_layers=conf.model['mlp_layers'],
                         dropout=conf.model['dropout'],
                         train_eps=conf.model['train_eps']).to(self.device)
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                        weight_decay=self.conf.training['weight_decay'])
        else:
            raise NotImplementedError

    def train(self):
        
        node_indices = np.arange(self.n_nodes)

        for epoch in range(self.conf.training['n_epochs']):
            # print(f'Epoch: {epoch}')
            self.model.train()
            self.optim.zero_grad()
            features, adj = self.feats, self.adj
            if self.method == 'mlp':
                output = self.model(features)
            else:
                output = self.model(features, adj)
            loss_train = self.loss_fn(output[node_indices], self.noisy_labels[node_indices])

            loss_train.backward()
            self.optim.step()

            # Record loss for each node
            with torch.no_grad():
                if self.method == 'mlp':
                    output = self.model(features)
                else:
                    output = self.model(features, adj)
                loss_per_node = F.cross_entropy(output, self.noisy_labels, reduction='none')
                self.loss_trajectories[:, epoch] = loss_per_node.cpu().numpy()

        return self.loss_trajectories
    


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str,
                        default='cornell',
                        choices=['cora_ml', 'wikics',  'products', 'children','history', 'photo',  'cornell', 'texas','washington', 'wisconsin'], 
                        help='Select dataset')
    parser.add_argument('--noise_type', type=str,
                        default='llm',
                        choices=['clean', 'uniform', 'pair', 'llm', 'topology', 'feature', 'confidence'], help='Type of label noise')
    parser.add_argument('--noise_rate', type=float,  default=None, help='Label noise rate')
    parser.add_argument('--method', type=str, default='sage', choices=['gcn', 'sage', 'gin', 'mlp', 'gat'], help="Select methods")
    
    
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--data_root', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--device', type=str, default='cuda', help='Device')


    args = parser.parse_args()
    return args


def run_GMM(losses, is_corrupted, random_state):
    gmm = GaussianMixture(n_components=2, random_state=random_state)
    gmm.fit(losses.reshape(-1, 1))
    gmm_labels = gmm.predict(losses.reshape(-1, 1))

    loss_label_zero = losses[gmm_labels == 0]
    loss_label_one = losses[gmm_labels == 1]

    if loss_label_zero.mean() > loss_label_one.mean():
        gmm_labels = 1 - gmm_labels

    roc_auc = roc_auc_score(is_corrupted, gmm_labels)
    return roc_auc * 100



def run_single_exp(dataset, args, debug=True):

    model_conf = load_conf(None, args.method, dataset.name)



    model_conf.training['n_epochs'] = 100
    model_conf.model['n_feat'] = dataset.dim_feats
    model_conf.model['n_classes'] =  dataset.n_classes 
    model_conf.training['debug'] = debug
    

    predictor = Predictor(model_conf, dataset, args.method, args.device, args.seed)

    loss_traj = predictor.train()
    
    is_corrupted = (dataset.noisy_labels != dataset.labels).cpu().numpy().astype(int)


    roc_auc_list = []
    for epoch in range(loss_traj.shape[1]):
        losses = loss_traj[:, epoch]
        roc = run_GMM(losses, is_corrupted, args.seed)
        
        roc_auc_list.append(roc)
    
    best_epoch = np.argmax(roc_auc_list)
    best_roc = roc_auc_list[best_epoch]

    avg_loss = np.mean(loss_traj, axis=1)
    roc_mean = run_GMM(avg_loss, is_corrupted, args.seed)

    return best_epoch, best_roc, roc_mean


if __name__ == '__main__':

    args = load_args()

    data_conf = load_conf('./config/datasets/' + args.data + '.yaml')
    
    best_epoch_list = []
    best_roc_list = []
    roc_mean_list = []

    for run, seed in enumerate(range(args.start_seed, args.runs + args.start_seed)):
        args.seed = seed
        setup_seed(args.seed)

        dataset = Dataset(name=args.data, conf=data_conf, noise_type=args.noise_type, path=args.data_root, device=args.device)

        best_epoch, best_roc, roc_mean = run_single_exp(dataset, args, debug=False)
        best_epoch_list.append(best_epoch)
        best_roc_list.append(best_roc)
        roc_mean_list.append(roc_mean)

    best_epoch_list = np.array(best_epoch_list)
    best_roc_list = np.array(best_roc_list)
    roc_mean_list = np.array(roc_mean_list)

    print(f'Best Epoch: {best_epoch_list.mean():.1f}')
    print(f'Best ROCAUC: {best_roc_list.mean():.1f}')
    print(f'Mean ROCAUC: {roc_mean_list.mean():.1f}')





