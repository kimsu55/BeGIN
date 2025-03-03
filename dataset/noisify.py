import numpy as np
import torch

import copy
from numpy.testing import assert_array_almost_equal
import torch.nn.functional as F
import torch_geometric.transforms as T
from tqdm import tqdm
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops, to_torch_csr_tensor
from dataset.utils import setup_seed



def get_diffusion_data(data, alpha=0.2, self_loop_weight=1, exact=True, norm_in='row', norm_out='row'):
    
    transform = T.GDC(
        self_loop_weight=self_loop_weight,
        normalization_in=norm_in, 
        normalization_out=norm_out,
        diffusion_kwargs=dict(method='ppr', alpha=alpha, eps=1e-6),
        exact=exact)
    return transform(data)




def calculate_transition_prob_vectorized(data, y, label_list, temp): 
    
    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y, dtype=torch.long)
    N = len(y)

    edge_attr = getattr(data, 'edge_attr', None)

    if edge_attr is None:
        edge_attr = torch.ones(data.edge_index.size(1), dtype=torch.float)


    new_edge_index, new_edge_attr = remove_self_loops(data.edge_index, edge_attr)
    new_edge_index = new_edge_index
    row_idx = new_edge_index[0] 
    col_idx = new_edge_index[1]

    adj_matrix = to_torch_csr_tensor(new_edge_index, new_edge_attr, size=(N, N))


    modified_labels = y + 1  # Shift labels to avoid 0 values
    edge_attr_label = modified_labels[col_idx]
    adj_matrix_label = to_torch_csr_tensor(new_edge_index, edge_attr_label, size=(N, N))

    transition_probs = torch.zeros((N, len(label_list)), dtype=torch.float)

     
    label_vals = adj_matrix_label.values()        
    edge_vals = adj_matrix.values()   


    for c in tqdm(label_list, desc='Computing transition probabilities'):
        c_mask = (label_vals == (c + 1))  # edges pointing to label c
        c_rows = row_idx[c_mask]         # source nodes for edges that point to label c
        c_edge_weights = edge_vals[c_mask]
        row_sums = scatter_add(c_edge_weights, c_rows, dim=0, dim_size=N)
        transition_probs[:, c] = row_sums


    row_sum = transition_probs.sum(dim=1, keepdim=True)   # shape: [N, 1]
    zero_mask = (row_sum.squeeze(1) == 0)   # shape: [N]
    

    logits = transition_probs / temp
    softmax_probs = F.softmax(logits, dim=1)

    softmax_probs[zero_mask] = 0
    softmax_probs[zero_mask, y[zero_mask]] = 1.0

    return softmax_probs.numpy()



def calculate_transition_matrix1(clean_labels, noisy_labels, num_classes): 
    """
    Calculate the transition matrix from clean labels to noisy labels using NumPy.
    
    Args:
        clean_labels (np.ndarray or torch.tensor): The true labels (clean labels).
        noisy_labels (np.ndarray or torch.tensor): The noisy labels.
        
    Returns:
        np.ndarray: Transition matrix of shape (num_classes, num_classes).
    """
    
    # Initialize the transition matrix
    transition_matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
    
    # Iterate through all label pairs
    for c, n in zip(clean_labels, noisy_labels):
        transition_matrix[c, n] += 1  # Increment the count for the (clean, noisy) pair
    
    # Normalize the rows to convert counts into probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).flatten()
    transition_matrix[zero_rows] = 1 / num_classes
    row_sums[row_sums == 0] = 1 
    transition_matrix = transition_matrix / row_sums
    
    return transition_matrix


def noisify_topology(data, name, num_classes, noise_rate, seed=0, diffusion_alpha=0.9, diffusion_exact=False, temp=0.01):


    y = data.y.numpy()
    noisy = y.copy()
    label_list = list(range(num_classes))
    num_data = y.shape[0]
    num_corruption = int(num_data * noise_rate)
    setup_seed(seed)


    if name in ['cora_ml', 'wickics', 'cornell', 'texas','washington', 'wisconsin' ]:
        diffusion_exact = True


    data = get_diffusion_data(copy.copy(data), alpha=diffusion_alpha, exact=diffusion_exact)
    softmax_labels_prob = calculate_transition_prob_vectorized(data, y, label_list, temp) # N x n_classes, N  
    labels_disagree_at_clean_labels = 1 - softmax_labels_prob[np.arange(num_data), y]

    labels_disagree_at_clean_labels = labels_disagree_at_clean_labels / labels_disagree_at_clean_labels.sum()


    indices = np.random.choice(y.shape[0], size=num_corruption, p=labels_disagree_at_clean_labels, replace=False)
    for i in tqdm(indices, desc='corrupting process'):
        true_label = y[i]
        modified_probs = softmax_labels_prob[i].copy()
        modified_probs[true_label] = 0
        modified_probs /= modified_probs.sum()  # Re-normalize probabilities
        noisy[i] = np.random.choice(num_classes, p=modified_probs)

    
    tm = calculate_transition_matrix1(y, noisy, num_classes)

    noisy = torch.tensor(noisy)
    
    return noisy, tm



def compute_class_features(X, labels, num_classes):
    class_sum = np.zeros((num_classes, X.shape[1]))
    for label in range(num_classes):
        class_sum[label] = np.sum(X[labels == label], axis=0)
    row_sums = np.sum(class_sum, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10
    class_feats = class_sum/row_sums
    return class_feats



def calculate_transition_prob_feature(node_feats, class_feats, labels, temp):


    epsilon = 1e-10
    node_norm = node_feats / (np.linalg.norm(node_feats, axis=1, keepdims=True) + epsilon)  # (N, d)
    class_norm = class_feats / (np.linalg.norm(class_feats, axis=1, keepdims=True) + epsilon) # (C, d)
    
    # Compute cosine similarity: dot product of normalized features
    cosine_sim = np.dot(node_norm, class_norm.T)  # (N, C)
    
    # Apply temperature-scaled softmax row-wise
    exp_scaled = np.exp(cosine_sim / temp)  # Scale similarities by temperature
    softmax_similarities = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)  # Softmax
    
    clean_probs = softmax_similarities[np.arange(len(labels)), labels]
    corrupt_probs = 1 - clean_probs
    return softmax_similarities, corrupt_probs


def noisify_feature(node_feats, labels, num_classes, noise_rate, temp=0.01, seed=0):
    setup_seed(seed)
    noisy_labels = labels.copy()
    num_corruption = int(node_feats.shape[0] * noise_rate)

    class_feats = compute_class_features(node_feats, labels, num_classes)


    transiton_probs, corrupt_probs = calculate_transition_prob_feature(node_feats, class_feats, labels, temp)

    corrupt_probs = corrupt_probs / corrupt_probs.sum()

    indices = np.random.choice(labels.shape[0], size=num_corruption, p=corrupt_probs, replace=False)

    for i in indices:
        true_label = labels[i]
        modified_probs = transiton_probs[i].copy()
        modified_probs[true_label] = 0
        modified_probs /= modified_probs.sum()  # Re-normalize probabilities
        noisy_labels[i] = np.random.choice(num_classes, p=modified_probs)

    
    tm = calculate_transition_matrix1(labels, noisy_labels, num_classes)

    noisy_labels = torch.tensor(noisy_labels)
    
    return noisy_labels, tm




def noisify_confidence(probs, labels, num_classes, noise_rate, seed=0, option='sample'):

    setup_seed(seed)
    noisy = labels.copy()
    num_corruption = int(probs.shape[0] * noise_rate)


    transiton_probs = probs/ probs.sum(axis=1, keepdims=True)
    corrupt_probs = 1 - transiton_probs[np.arange(len(labels)), labels]

    corrupt_probs = corrupt_probs / corrupt_probs.sum()

    if option == 'sample':
        indices = np.random.choice(labels.shape[0], size=num_corruption, p=corrupt_probs, replace=False)
    elif option == 'hard':
        indices = np.argsort(corrupt_probs)[:num_corruption]

    for i in indices:
        true_label = labels[i]
        modified_probs = transiton_probs[i].copy()
        modified_probs[true_label] = 0
        modified_probs /= modified_probs.sum()  # Re-normalize probabilities
        noisy[i] = np.random.choice(num_classes, p=modified_probs)

    noisy = torch.tensor(noisy)
    tm = calculate_transition_matrix1(labels, noisy, num_classes)
    
    return noisy, tm


def uniform_noise_tm(n_classes, noise_rate):
    P = np.float64(noise_rate) / np.float64(n_classes - 1) * np.ones((n_classes, n_classes))
    np.fill_diagonal(P, (np.float64(1) - np.float64(noise_rate)) * np.ones(n_classes))
    diag_idx = np.arange(n_classes)
    P[diag_idx, diag_idx] = P[diag_idx, diag_idx] + 1.0 - P.sum(0)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def pair_noise_tm(n_classes, noise_rate):
    P = (1.0 - np.float64(noise_rate)) * np.eye(n_classes)
    for i in range(n_classes):
        P[i, i - 1] = np.float64(noise_rate)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def random_noise_tm(n_classes, noise_rate):
    P = (1.0 - np.float64(noise_rate)) * np.eye(n_classes)
    for i in range(n_classes):
        tp = np.random.rand(n_classes)
        tp[i] = 0
        tp = (tp / tp.sum()) * noise_rate
        P[i, :] += tp
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def add_label_noise(labels, cp, random_seed):
    assert_array_almost_equal(cp.sum(axis=1), np.ones(cp.shape[1]))
    n_labels = labels.shape[0]
    noisy_labels = labels.copy()
    rs = np.random.RandomState(random_seed)

    for i in range(n_labels):
        label = labels[i]
        flipped = rs.multinomial(1, cp[label, :], 1)[0]
        noisy_label = np.where(flipped == 1)[0]
        noisy_labels[i] = noisy_label

    return noisy_labels


def noisify_class_dependent(labels, num_classes, noise_type='uniform', noise_rate=0.2, random_seed=5):

    setup_seed(random_seed)
    assert (noise_rate >= 0.) and (noise_rate <= 1.)
    if noise_type == 'uniform':
        tm = uniform_noise_tm(num_classes, noise_rate)
    elif noise_type == 'random':
        tm = random_noise_tm(num_classes, noise_rate)
    elif noise_type == 'pairwise':
        tm = pair_noise_tm(num_classes, noise_rate)
    else:
        raise ValueError('Invalid noise type')

    noisy_labels = add_label_noise(labels, tm, random_seed)
    noisy_labels = torch.tensor(noisy_labels)
    
   
    return noisy_labels,  tm



def noisify_dataset(dataset, noise_type, noise_rate=None, random_seed=0):
    """
    Add noise to the dataset.

    Args:
        dataset: a list of tuples (x, y) where x is a numpy array and y is a scalar.
        noise_rate: the rate of noise to add to the dataset.

    Returns:
        The noisy dataset.
    """

    num_classes = dataset.num_classes
    data = dataset[0]
    y = data.y
    y_np = y.numpy()
   
    if noise_rate == 0 or noise_type == 'clean':
        return copy.copy(y), None


    if noise_rate == None:
        llm_noise = data.noisy
        noise_rate = (llm_noise.numpy() != y_np).sum() / len(y)


    if noise_type == 'uniform':
        noisy_y, tm = noisify_class_dependent(y_np, num_classes, noise_type='uniform', noise_rate=noise_rate, random_seed=random_seed)
    elif noise_type == 'pairwise':
        noisy_y, tm = noisify_class_dependent(y_np, num_classes, noise_type='pairwise', noise_rate=noise_rate, random_seed=random_seed)
    
    elif noise_type == 'feature':
        node_feats = data.x.numpy()
        noisy_y, tm = noisify_feature(node_feats, y_np, num_classes, noise_rate, temp=0.01, seed=random_seed)
    
    elif noise_type == 'topology':
        noisy_y, tm = noisify_topology(data, dataset.name, num_classes, noise_rate, seed=random_seed)

    elif noise_type == 'confidence':
        model_prediction = np.load(f'dataset/model_prediction/{dataset.name}_sage.npz')
        probs = model_prediction['probs']
        model_labels = model_prediction['labels']
        assert np.all(model_labels == y_np)
        noisy_y, tm = noisify_confidence(probs, y_np, num_classes, noise_rate, seed=random_seed)
    
    elif noise_type == 'llm':
        noisy_y = data.noisy
        tm = calculate_transition_matrix1(y, noisy_y, num_classes)

    else:
        raise ValueError('Invalid noise type')
    
    
    actual_noise_rate = (noisy_y.numpy() != y_np).sum() / len(y)
    print(f'Actual noise rate of {noise_type}: {actual_noise_rate:.3f}')
    
        
    return noisy_y, tm



