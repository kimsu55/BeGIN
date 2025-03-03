import os.path as osp
import torch
import random
import numpy as np
from huggingface_hub import hf_hub_download
import shutil
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import torch.nn.functional as F

def download_hf_file(repo_id,
                     filename,
                     local_dir,
                     subfolder=None,
                     repo_type="dataset",
                     cache_dir=None,
                     ):

    hf_hub_download(repo_id=repo_id, subfolder=subfolder, filename=filename, repo_type=repo_type,
                    local_dir=local_dir,  cache_dir=cache_dir, force_download=True)
    if subfolder is not None:
        shutil.move(osp.join(local_dir, subfolder, filename), osp.join(local_dir, filename))
        shutil.rmtree(osp.join(local_dir, subfolder))
    return osp.join(local_dir, filename)


def generate_node_features(text_list, num_bags=1703):
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    """
    Generate node features using a Bag-of-Words (BoW) representation.

    Parameters:
        text_list (list of str): A list of textual data (e.g., web page content).
        num_bags (int): The maximum number of words in the vocabulary (default: 1703).

    Returns:
        node_features (array): A binary Bag-of-Words matrix (0 or 1),
                               where each row represents a text and each column represents a word in the vocabulary.
    """
    # Define stopwords
    stop_words = set(stopwords.words('english'))

    # Preprocess each text: tokenize, remove stopwords, and keep only alphabetic tokens
    def preprocess_text(text):
        tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
        return ' '.join([word for word in tokens if word.isalpha() and word not in stop_words])
        # return ' '.join([word for word in tokens if word.isalpha()])

    text_cleaned = [preprocess_text(text) for text in text_list]

    # Create a CountVectorizer for binary Bag-of-Words representation
    vectorizer = CountVectorizer(max_features=num_bags, binary=True)

    # Fit and transform the cleaned text data
    node_features = vectorizer.fit_transform(text_cleaned).toarray()

    return node_features


def setup_seed(seed):
    '''
    Setup random seed so that the experimental results are reproducible
    Parameters
    ----------
    seed : int
        random seed for torch, numpy and random

    Returns
    -------
    None
    '''
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def normalize(mx, style='symmetric', add_loop=True, p=None):
    '''
    Normalize the feature matrix or adj matrix.

    Parameters
    ----------
    mx : torch.tensor
        Feature matrix or adj matrix to normalize. Note that either sparse or dense form is supported.
    style: str
        If set as ``row``, `mx` will be row-wise normalized.
        If set as ``symmetric``, `mx` will be normalized as in GCN.
        If set as ``softmax``, `mx` will be normalized using softmax.
        If set as ``row-norm``, `mx` will be normalized using `F.normalize` in pytorch.
    add_loop : bool
        Whether to add self loop.
    p : float
        The exponent value in the norm formulation. Onlu used when style is set as ``row-norm``.
    Returns
    -------
    normalized_mx : torch.tensor
        The normalized matrix.
    '''
    if style == 'row':
        if mx.is_sparse:
            return row_normalize_sp(mx)
        else:
            return row_nomalize(mx)
    elif style == 'symmetric':
        if mx.is_sparse:
            return normalize_sp_tensor_tractable(mx, add_loop)
        else:
            return normalize_tensor(mx, add_loop)
    elif style == 'softmax':
        if mx.is_sparse:
            return torch.sparse.softmax(mx, dim=-1)
        else:
            return F.softmax(mx, dim=-1)
    elif style == 'row-norm':
        assert p is not None
        if mx.is_sparse:
            # TODO
            pass
        else:
            return F.normalize(mx, dim=-1, p=p)
    else:
        raise KeyError("The normalize style is not provided.")



def row_nomalize(mx):
    """Row-normalize sparse matrix.
    """

    r_sum = mx.sum(1)
    r_inv = r_sum.pow(-1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx

    return mx


def row_normalize_sp(mx):
    adj = mx.coalesce()
    inv_sqrt_degree = 1. / (torch.sparse.sum(mx, dim=1).values() + 1e-12)
    D_value = inv_sqrt_degree[adj.indices()[0]]
    new_values = adj.values() * D_value
    return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())


def normalize_sp_tensor_tractable(adj, add_loop=True):
    n = adj.shape[0]
    device = adj.device
    if add_loop:
        adj = adj + torch.eye(n, device=device).to_sparse()
    adj = adj.coalesce()
    inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()) + 1e-12)
    D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]
    new_values = adj.values() * D_value
    return torch.sparse_coo_tensor(adj.indices(), new_values, adj.size())


def normalize_tensor(adj, add_loop=True):
    device = adj.device
    adj_loop = adj + torch.eye(adj.shape[0]).to(device) if add_loop else adj
    rowsum = adj_loop.sum(1)
    r_inv = rowsum.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    A = r_mat_inv @ adj_loop
    A = A @ r_mat_inv
    return A



def sample_per_class(labels, num_examples_per_class, forbidden_indices=None):
    num_samples = len(labels)
    num_classes = labels.max() + 1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [np.random.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])

def get_split(labels, train_examples_per_class=None, val_examples_per_class=None, test_examples_per_class=None,
              train_size=None, val_size=None, test_size=None):
    num_samples = len(labels)
    num_classes = labels.max() + 1
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = np.random.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = np.random.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))

    if test_examples_per_class is not None:
        test_indices = sample_per_class(labels, test_examples_per_class, forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = np.random.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    return train_indices, val_indices, test_indices