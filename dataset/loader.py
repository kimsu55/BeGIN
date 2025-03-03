import torch

import numpy as np
from torch_geometric.utils import degree
import torch.nn.functional as F
import os
from torch_sparse.tensor import SparseTensor
from torch_geometric.utils import to_torch_csr_tensor
import csv
import pandas as pd
from dataset.utils import normalize, get_split
from dataset.BeGIN_dataset import NoisyGraphDataset, NoisyProducts, NoisyCoraML, NoisyWikiCS
from dataset.noisify import noisify_dataset



class NoisyData:
    '''
    Dataset Class.
    This class loads, preprocesses and splits various datasets.

    Parameters
    ----------
    name : str
        The name of dataset.
    conf : argparse.Namespace
        The configuration file.
    path : str
        Path to save dataset files.
    device : str
        The device to run the model.
    verbose : bool
        Whether to print statistics.
    '''

    def __init__(self, name,  conf, noise_type, noise_rate,  seed=0, path='./data/',  device='cuda:0',  verbose=False):
        self.name = name
        self.path = path
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.device = torch.device(device)
        self.seed = seed
        
        self.split_type = conf.split['split_type']
        self.train_size = conf.split['train_size']
        self.val_size = conf.split['val_size']
        self.test_size = conf.split['test_size']
        self.train_percent = conf.split['train_percent']
        self.val_percent = conf.split['val_percent']
        self.test_percent = conf.split['test_percent']
        self.train_examples_per_class = conf.split['train_examples_per_class']
        self.val_examples_per_class = conf.split['val_examples_per_class']
        self.test_examples_per_class = conf.split['test_examples_per_class']

        self.prepare_data(name, conf.norm['feat_norm'])
        self.split_data(verbose)
        
        if conf.modify['add_self_loop']:
            self.adj = self.adj + torch.eye(self.adj.shape[0], device=self.adj.device).to_sparse()
        if conf.norm['adj_norm']:
            self.adj = normalize(self.adj, add_loop=False)
        # self.adj = self.adj.coalesce()


    def prepare_data(self, name, feat_norm):

        if name == 'cora_ml':
            dataset = NoisyCoraML(root=self.path)
        elif name == 'products':
            dataset = NoisyProducts(root=self.path)
        elif name == 'wiki_cs':
            dataset = NoisyWikiCS(root=self.path)
        elif name in ['children','history', 'photo',  'cornell', 'texas','washington', 'wisconsin']:
            dataset = NoisyGraphDataset(root=self.path, name=name)

        self.g = dataset[0]
        self.n_edges = int(self.g.edge_index.shape[1] / 2)
        self.feats = self.g.x.to(torch.float)
        self.labels = self.g.y
        self.n_classes = self.labels.max().item() + 1
        self.n_nodes = self.feats.shape[0]
        self.dim_feats = self.feats.shape[1]
        self.adj = torch.sparse_coo_tensor(self.g.edge_index, torch.ones(self.g.edge_index.shape[1]), [self.n_nodes, self.n_nodes])
        self.adj = self.adj.to(torch.float)

        self.feats = self.feats.to(self.device)
        self.labels = self.labels.to(self.device)
        self.adj = self.adj.to(self.device)
        if feat_norm:  # 
            self.feats = normalize(self.feats, style='row')
        
        self.adj = self.adj.coalesce()
        self.adj_csr = self.adj.to_sparse_csr()

        self.noisy_label, tm = noisify_dataset(dataset, self.noise_type, self.noise_rate, random_seed=self.seed)
        self.noisy_label = self.noisy_label.to(self.device)
    
    
    
    def split_data(self, verbose=False):

        '''
        Function to conduct data splitting for various datasets.

        Parameters
        ----------
        verbose : bool
            Whether to print statistics.
        '''

        self.train_masks = None
        self.val_masks = None
        self.test_masks = None
        train_type = None
        val_type = None
        test_type = None

        if self.split_type == 'default':
            if not hasattr(self.g, 'train_mask'):
                print('Split error, split type=' + self.split_type + '. Dataset ' + self.name + ' has no default split')
                exit(0)
            train_indices = torch.nonzero(self.g.train_mask, as_tuple=False).squeeze().numpy()
            val_indices = torch.nonzero(self.g.val_mask, as_tuple=False).squeeze().numpy()
            test_indices = torch.nonzero(self.g.test_mask, as_tuple=False).squeeze().numpy()
            train_type = 'default'
            val_type = 'default'
            test_type = 'default'
        elif self.split_type == 'percent':
            if self.train_size is not None:
                train_size = self.train_size
                train_type = 'specified'
            elif self.train_percent is not None:
                train_size = int(self.n_nodes * self.train_percent)
                train_type = str(self.train_percent * 100) + ' % of nodes'
            else:
                print('Split error: split type = percent. Train size and train percent were not configured')
                exit(0)

            if self.val_size is not None:
                val_size = self.val_size
                val_type = 'specified'
            elif self.val_percent is not None:
                val_size = int(self.n_nodes * self.val_percent)
                val_type = str(self.val_percent * 100) + ' % of nodes'
            else:
                print('Split error: split type = percent. Val size and Val percent were not configured')
                exit(0)

            if self.test_size is not None:
                test_size = self.test_size
                test_type = 'specified'
            elif self.test_percent is not None:
                test_size = int(self.n_nodes * self.test_percent)
                test_type = str(self.test_percent * 100) + ' % of nodes'
            else:
                test_size = None
                test_type = 'remaining'
            train_indices, val_indices, test_indices = get_split(self.labels.cpu().numpy(),
                                                                 train_size=train_size,
                                                                 val_size=val_size,
                                                                 test_size=test_size, )
        elif self.split_type == 'samples_per_class':
            train_size = None
            val_size = None
            test_size = None
            if self.train_examples_per_class is not None:
                train_examples_per_class = self.train_examples_per_class
                train_type = str(self.train_examples_per_class) + ' nodes per class'
            elif self.train_size is not None:
                train_examples_per_class = None
                train_size = self.train_size
                train_type = 'specified'
            else:
                print('Split error: split type = samples_per_class. Train size and train percent were not configured')
                exit(0)

            if self.val_examples_per_class is not None:
                val_examples_per_class = self.val_examples_per_class
                val_type = str(self.val_examples_per_class) + ' nodes per class'
            elif self.val_size is not None:
                val_examples_per_class = None
                val_size = self.val_size
                val_type = 'specified'
            else:
                print('Split error: split type = samples_per_class. Val size and val percent were not configured')
                exit(0)

            if self.test_examples_per_class is not None:
                test_examples_per_class = self.test_examples_per_class
                test_type = str(self.test_examples_per_class) + ' nodes per class'
            elif self.test_size is not None:
                test_examples_per_class = None
                test_size = self.test_size
                test_type = 'specified'
            else:
                test_examples_per_class = None
                test_size = None
                test_type = 'remaining'
            train_indices, val_indices, test_indices = get_split(self.labels.cpu().numpy(),
                                                                 train_examples_per_class=train_examples_per_class,
                                                                 val_examples_per_class=val_examples_per_class,
                                                                 test_examples_per_class=test_examples_per_class,
                                                                 train_size=train_size,
                                                                 val_size=val_size,
                                                                 test_size=test_size)
        else:
            print('Split error: split type ' + self.split_type + ' not implemented')
            exit(0)

        self.train_masks = train_indices
        self.val_masks = val_indices
        self.test_masks = test_indices

        if verbose:
            print("""----Split statistics------'
                #Train samples %d (%s)
                #Val samples %d (%s)
                #Test samples %d (%s)""" %
                  (len(self.train_masks), train_type,
                   len(self.val_masks), val_type,
                   len(self.test_masks), test_type))





