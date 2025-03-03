from copy import deepcopy

import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn.functional as F
from models.GNNs import GCN, GraphSAGE, MLP, GAT, GIN
from models.tools import Recorder, accuracy

class BasePredictor:
    def __init__(self, conf, dataset, method,  device='cuda:0', seed=0):
        super(BasePredictor, self).__init__()
        self.conf = conf
        self.device = torch.device(device)
        self.seed = seed
        self.method = method
        self.general_init(dataset)
        self.method_init(conf)

    def general_init(self, dataset):
        '''
        This conducts necessary operations for an experiment, including the setting specified split,
        variables to record statistics.
        '''
        self.loss_fn = F.binary_cross_entropy_with_logits if dataset.n_classes == 1 else F.cross_entropy
        # self.edge_index = dataset.adj.indices()
        self.adj = dataset.adj if self.conf.dataset['sparse'] else dataset.adj.to_dense()
        self.feats = dataset.feats
        self.n_nodes = dataset.n_nodes
        self.n_classes = dataset.n_classes
        self.noisy_labels = dataset.noisy_labels
        self.conf.model['n_feat'] = dataset.dim_feats
        self.conf.model['n_classes'] =  dataset.n_classes

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


class Detector(BasePredictor):
    def __init__(self, conf, dataset, method,  device='cuda:0', seed=0):

    
        super().__init__(conf, dataset, method,  device, seed)

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
    


class NodeClassifier(BasePredictor):
    def __init__(self, conf, dataset, method,  device='cuda:0', seed=0):
                
        self.clean_label = dataset.labels
        self.metric = roc_auc_score if dataset.n_classes == 1 else accuracy
        self.recoder = Recorder(conf.training['patience'], conf.training['criterion'])
        self.train_mask = dataset.train_masks
        self.val_mask = dataset.val_masks
        self.test_mask = dataset.test_masks
        self.result = {'test_loss':-1, 'train': -1, 'valid': -1, 'test': -1}
        
        super().__init__(conf, dataset, method,  device, seed)
    
    
    def get_prediction(self, features, adj, label=None, mask=None):
        
        if self.method == 'mlp':
            output = self.model(features)
        else:
            output = self.model(features, adj)
        loss, acc = None, None
        if (label is not None) and (mask is not None):
            if self.n_classes == 1:
                output = output.squeeze()
            loss = self.loss_fn(output[mask], label[mask])

            acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
        return output, loss, acc

    def train(self):
        '''
        This is the common training procedure, which is overwritten for special learning procedure.

        Parameters
        ----------
        None

        Returns
        -------
        result : dict
            A dict containing train, valid and test metrics.
        '''

        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            self.model.train()
            self.optim.zero_grad()
            features, adj = self.feats, self.adj
            # forward and backward
            output, loss_train, acc_train = self.get_prediction(features, adj, self.noisy_labels, self.train_mask)
            loss_train.backward()
            self.optim.step()

            # Evaluate
            loss_val, acc_val = self.evaluate(self.noisy_labels, self.val_mask)
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
            if flag:
                self.best_val_loss = loss_val
                self.result['valid'] = round(acc_val * 100, 2) 
                self.result['train'] = round(acc_train *100, 2)
                self.weights = deepcopy(self.model.state_dict())
            elif flag_earlystop:
                break


        loss_test, acc_test = self.test(self.test_mask)
        self.result['test'] = round(acc_test*100, 2)
        self.result['test_loss'] = loss_test.item()
        return self.result


    def evaluate(self, label, mask):
        '''
        This is the common evaluation procedure, which is overwritten for special evaluation procedure.

        Parameters
        ----------
        label : torch.tensor
        mask: torch.tensor

        Returns
        -------
        loss : float
            Evaluation loss.
        metric : float
            Evaluation metric.
        '''
        self.model.eval()
        features, adj = self.feats, self.adj
        with torch.no_grad():
            _, loss, acc = self.get_prediction(features, adj, label, mask)
        return loss, acc

    def test(self, mask):
        '''
        This is the common test procedure, which is overwritten for special test procedure.

        Returns
        -------
        loss : float
            Test loss.
        metric : float
            Test metric.
        '''
        if self.weights is not None:
            self.model.load_state_dict(self.weights)
        label = self.clean_label
        return self.evaluate(label, mask)