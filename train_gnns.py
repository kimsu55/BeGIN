
import argparse
import numpy as np
from dataset.utils import setup_seed
from dataset.loader import NoisyData
from models.gnn_predictor import NodeClassifier
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
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--data_root', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    return args


def run_single_exp(dataset, args):

    model_conf = load_conf(None, args.method, dataset.name)
    predictor = NodeClassifier(model_conf, dataset, args.method, args.device, args.seed)
    results = predictor.train()

    return results


if __name__ == '__main__':

    args = load_args()
    data_conf = load_conf('./config/datasets/' + args.data + '.yaml')
    
    test_losses = []
    train_acc = []
    valid_acc = []
    test_acc = []

    for run, seed in enumerate(range(args.start_seed, args.runs + args.start_seed)):
        args.seed = seed
        setup_seed(args.seed)

        dataset = NoisyData(name=args.data, conf=data_conf, noise_type=args.noise_type, noise_rate=args.noise_rate, seed=args.seed, path=args.data_root, device=args.device)

        results = run_single_exp(dataset, args)
        
        test_losses.append(results['test_loss'])
        train_acc.append(results['train'])
        valid_acc.append(results['valid'])
        test_acc.append(results['test'])
        print(f'Run {run+1}/{args.runs} finished: Test Loss: {results["test_loss"]:.4f}, Train Acc: {results["train"]}, Valid Acc: {results["valid"]}, Test Acc: {results["test"]}')
    
    print(f'Avg Test Loss: {np.mean(test_losses):.4f}, Avg Train Acc: {np.mean(train_acc):.2f}, Avg Valid Acc: {np.mean(valid_acc):.2f}, Avg Test Acc: {np.mean(test_acc):.2f}')





