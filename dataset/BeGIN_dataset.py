
import ast
import pandas as pd
import numpy as np
import os.path as osp
import json
import warnings
from itertools import chain
from typing import Optional, Callable, List, Optional

import torch
from torch_geometric.data import InMemoryDataset, Data, download_url
from torch_geometric.utils import sort_edge_index, to_edge_index, to_undirected
from torch_geometric.utils.isolated import remove_self_loops
from torch_geometric.io import read_npz

from dataset.utils import generate_node_features, download_hf_file



class NoisyGraphDataset(InMemoryDataset):
    """
    PyG Dataset for BeGIN.

    Args:
        root (str): Root directory where the dataset is stored.
        dataset_name (str): Name of the dataset (e.g., "children", "Cornell").
        transform (callable, optional): Data transformation function.
        pre_transform (callable, optional): Preprocessing function before saving.
    """
    def __init__( self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name
        assert self.name in ['children', 'photo', 'history', 'cornell', 'texas','washington', 'wisconsin']

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')


    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')


    @property
    def raw_file_names(self):
        files  = [f"{self.name.title()}.csv", 'noisy_labels.csv']
        if self.name in ["children", "photo", "history"]:
            files += [f"{self.name.title()}_roberta_base_512_cls.npy"]
        return files

    @property
    def processed_file_names(self):
        return 'data.pt'


    def download(self):
        for file in self.raw_file_names:
            download_hf_file(repo_id="kimsu55/BeGIN", subfolder=self.name, repo_type="dataset", filename=file, local_dir=self.raw_dir)


    def process(self):

        df = pd.read_csv(self.raw_paths[0], sep=',', header=0)
        neighbor_id = list(df['neighbor_ids'])
        label = torch.tensor(df['label'].to_numpy(), dtype=torch.long)
        node_id = df['node_id'].to_numpy()
        num_nodes = len(node_id)
        if self.name in ['children', 'history', 'photo']:
            feats = torch.from_numpy(np.load(self.raw_paths[-1]).astype(np.float32))
        elif self.name in ['cornell', 'texas', 'washington', 'wisconsin']: 
            text = df['text'].to_numpy()
            feats = generate_node_features(text, num_bags=1703)

        df_n = pd.read_csv(self.raw_paths[1])
        true_labels = torch.tensor(df_n['True_labels'].to_numpy(), dtype=torch.long)
        llm_noisy = torch.tensor(df_n['Aggre'].to_numpy(), dtype=torch.long)

        assert torch.all(label == true_labels)

        source_nodes = []
        target_nodes = []

        for src, neighbors in zip(node_id, neighbor_id):
            neighbors = ast.literal_eval(neighbors)  # Convert string to list
            if isinstance(neighbors, list):  # Ensure it's a valid list
                # print(i, src, neighbors)
                for tgt in neighbors:
                    source_nodes.append(src)  # Add source node
                    target_nodes.append(tgt)  # Add target node
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = sort_edge_index(edge_index)
        edge_index =  to_undirected(edge_index)

        data = Data(x=feats, edge_index=edge_index, y=label, noisy=llm_noisy)

        if self.name in ['photo']:
            year = df['year'].to_list()
            np.random.seed(42)
            train_year = 2015
            val_year = 2016 
            indices = np.arange(num_nodes)
            valid_indices = [i for i in indices if label[i].item() != -1]
            train_ids = [i for i in valid_indices if year[i] < train_year]
            val_ids = [i for i in valid_indices if year[i] >= train_year and year[i] < val_year]
            test_ids = [i for i in valid_indices if year[i] >= val_year]
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[train_ids] = True
            data.train_mask = train_mask
            
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask[val_ids] = True
            data.val_mask = val_mask
            
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask[test_ids] = True
            data.test_mask = test_mask

        # Apply pre-transform if available
        if self.pre_transform:
            data = self.pre_transform(data)

        # Save processed dataset
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'Noisy-{self.name.capitalize()}'




class NoisyProducts(InMemoryDataset):
    r"""This is a subset of a co-purchase network from the Amazon platform. Nodes represent the products sold on Amazon and edges represent two products that are co-purchased together.
    The dataset originates from the `"Open Graph Benchmark: Datasets for Machine Learning on Graphs", and a subset has been curated from the
    `"HARNESSING EXPLANATIONS: LLM-TO-LM INTERPRETER FOR ENHANCED TEXT-ATTRIBUTED GRAPH REPRESENTATION LEARNING"
    The code references `"TAGLAS: An atlas of text-attributed graph datasets in the era of large graph and language models"

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        is_undirected (bool, optional): Whether the graph is undirected.
            (default: :obj:`True`)
    """

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        
        self.name = "products"
        save_dir = osp.join(root, "products")
        super().__init__(save_dir, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["ogbn-products_subset.pt", "noisy_labels.csv"]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_hf_file(repo_id="WFRaain/TAG_datasets", subfolder="products", filename="ogbn-products_subset.pt", local_dir=self.raw_dir)
        download_hf_file(repo_id="kimsu55/BeGIN", subfolder="products", repo_type="dataset", filename="noisy_labels.csv", local_dir=self.raw_dir)

    def process(self):
        data = torch.load(self.raw_paths[0])

        x = data.x.to(torch.float)
        y = data.y.squeeze().to(torch.long)

        edge_index = data.adj_t.to_symmetric()
        edge_index = to_edge_index(edge_index)[0]
        train_mask, val_mask, test_mask = data.train_mask.squeeze(), data.val_mask.squeeze(), data.test_mask.squeeze()

        df_n = pd.read_csv(self.raw_paths[1])
        true_labels = torch.tensor(df_n['True_labels'].to_numpy(), dtype=torch.long)
        llm_noisy = torch.tensor(df_n['Aggre'].to_numpy(), dtype=torch.long)

        assert torch.all(y == true_labels)



        data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask, noisy=llm_noisy)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return 'Noisy-Products_subset'




class NoisyCoraML(InMemoryDataset):
    r"""The full citation network datasets from the
    `"Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via
    Ranking" <https://arxiv.org/abs/1707.03815>`_ paper.
    Nodes represent documents and edges represent citation links.
    Datasets include :obj:`"Cora"`, :obj:`"Cora_ML"`, :obj:`"CiteSeer"`,
    :obj:`"DBLP"`, :obj:`"PubMed"`.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"Cora"`, :obj:`"Cora_ML"`
            :obj:`"CiteSeer"`, :obj:`"DBLP"`, :obj:`"PubMed"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        to_undirected (bool, optional): Whether the original graph is
            converted to an undirected one. (default: :obj:`True`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    * - Cora_ML
        - #nodes: 2,995
        - #edges: 16,316
        - #features: 2,879
        - #classes: 7


    """

    url = 'https://github.com/abojchevski/graph2gauss/raw/master/data/{}.npz'

    def __init__(self, root: str, 
                 transform = None,
                 pre_transform = None):
        super().__init__(root, transform, pre_transform)
        self.name = 'cora_ml'
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'cora_ml', 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'cora_ml', 'processed')

    @property
    def raw_file_names(self) -> str:
        return ['cora_ml.npz', 'noisy_labels.csv']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_hf_file(repo_id="kimsu55/BeGIN", subfolder="cora_ml", repo_type="dataset", filename="noisy_labels.csv", local_dir=self.raw_dir)
        download_url(self.url.format('cora_ml'), self.raw_dir)

    def process(self):
        data = read_npz(self.raw_paths[0])

        df_n = pd.read_csv(self.raw_paths[1])
        true_labels = torch.tensor(df_n['True_labels'].to_numpy(), dtype=torch.long)
        llm_noisy = torch.tensor(df_n['Aggre'].to_numpy(), dtype=torch.long)

        assert torch.all(data.y == true_labels)
        data.noisy = llm_noisy

        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return 'Noisy-Cora_ML'



class NoisyWikiCS(InMemoryDataset):
    r"""The semi-supervised Wikipedia-based dataset from the
    `"Wiki-CS: A Wikipedia-Based Benchmark for Graph Neural Networks"
    <https://arxiv.org/abs/2007.02901>`_ paper, containing 11,701 nodes,
    216,123 edges, 10 classes and 20 different training splits.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        is_undirected (bool, optional): Whether the graph is undirected.
            (default: :obj:`True`)
    """

    url = 'https://github.com/pmernyei/wiki-cs-dataset/raw/master/dataset'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 is_undirected: Optional[bool] = None):
        self.name = 'wikics'
        if is_undirected is None:
            warnings.warn(
                f"The {self.__class__.__name__} dataset now returns an "
                f"undirected graph by default. Please explicitly specify "
                f"'is_undirected=False' to restore the old behavior.")
            is_undirected = True
        self.is_undirected = is_undirected
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['data.json', 'noisy_labels.csv']

    @property
    def processed_file_names(self) -> str:
        return 'data_undirected.pt' if self.is_undirected else 'data.pt'

    def download(self):
        download_hf_file(repo_id="kimsu55/BeGIN", subfolder="wikics", repo_type="dataset", filename="noisy_labels.csv", local_dir=self.raw_dir)
        for name in self.raw_file_names:
            download_url(f'{self.url}/{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = json.load(f)

        x = torch.tensor(data['features'], dtype=torch.float)
        y = torch.tensor(data['labels'], dtype=torch.long)

        edges = [[(i, j) for j in js] for i, js in enumerate(data['links'])]
        edges = list(chain(*edges))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        if self.is_undirected:
            edge_index = to_undirected(edge_index, num_nodes=x.size(0))

        train_mask = torch.tensor(data['train_masks'], dtype=torch.bool)
        train_mask = train_mask.t().contiguous()

        val_mask = torch.tensor(data['val_masks'], dtype=torch.bool)
        val_mask = val_mask.t().contiguous()

        test_mask = torch.tensor(data['test_mask'], dtype=torch.bool)

        stopping_mask = torch.tensor(data['stopping_masks'], dtype=torch.bool)
        stopping_mask = stopping_mask.t().contiguous()

        df_n = pd.read_csv(self.raw_paths[1])
        true_labels = torch.tensor(df_n['True_labels'].to_numpy(), dtype=torch.long)
        llm_noisy = torch.tensor(df_n['Aggre'].to_numpy(), dtype=torch.long)

        assert torch.all(y == true_labels)

        data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask,
                    stopping_mask=stopping_mask, noisy=llm_noisy)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


    def __repr__(self) -> str:
        return 'Noisy-WikiCS'