
import argparse

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

from dataset.utils import setup_seed
from dataset.loader import Dataset
from models.predictor import Detector
from models.tools import load_conf


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str,
                        default='cornell',
                        choices=['cora_ml', 'wikics',  'products', 'children','history', 'photo',  'cornell', 'texas','washington', 'wisconsin'], 
                        help='Select dataset')
    parser.add_argument('--noise_type', type=str,
                        default='llm',
                        choices=['clean', 'uniform', 'pair', 'llm', 'topology', 'feature', 'confidence'], help='Type of label noise')
    parser.add_argument('--noise_rate', type=float,  default=None, help='Label noise rate, If set to None, the noise rate will be automatically derived from the LLM-based label noise in the dataset.')
    parser.add_argument('--method', type=str, default='sage', choices=['gcn', 'sage', 'gin', 'mlp', 'gat'], help="Select methods")
    
    
    parser.add_argument('--runs', type=int, default=2)
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



def run_single_exp(dataset, args):

    model_conf = load_conf(None, args.method, dataset.name)
    model_conf.training['n_epochs'] = 100
    
    predictor = Detector(model_conf, dataset, args.method, args.device, args.seed)
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

        best_epoch, best_roc, roc_mean = run_single_exp(dataset, args)
        best_epoch_list.append(best_epoch)
        best_roc_list.append(best_roc)
        roc_mean_list.append(roc_mean)
        print(f'Run {run+1}/{args.runs} finished: Best Epoch: {best_epoch}, Best ROCAUC: {best_roc:.2f}, Mean  ROCAUC: {roc_mean:.2f}')

    best_epoch_list = np.array(best_epoch_list)
    best_roc_list = np.array(best_roc_list)
    roc_mean_list = np.array(roc_mean_list)


    print(f'Avg Best Epoch: {best_epoch_list.mean():.2f}, Avg Best ROCAUC: {best_roc_list.mean():.2f}, Avg Mean ROCAUC: {roc_mean_list.mean():.2f}')





