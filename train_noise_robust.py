import argparse
import numpy as np
from dataset.utils import setup_seed
from dataset.loader import NoisyData
from models.tools import load_conf
import sys
import os
# Dynamically add NoisyGL to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "NoisyGL")))
from NoisyGL.predictor.LCAT_Predictor import lcat_Predictor
from NoisyGL.predictor.Smodel_Predictor import smodel_Predictor
from NoisyGL.predictor.Forward_Predictor import forward_Predictor
from NoisyGL.predictor.Backward_Predictor import backward_Predictor
from NoisyGL.predictor.Coteaching_Predictor import coteaching_Predictor
from NoisyGL.predictor.SCE_Predictor import sce_Predictor
from NoisyGL.predictor.JoCoR_Predictor import jocor_Predictor
from NoisyGL.predictor.APL_Predictor import apl_Predictor
from NoisyGL.predictor.DGNN_Predictor import dgnn_Predictor
from NoisyGL.predictor.CP_Predictor import cp_Predictor
from NoisyGL.predictor.NRGNN_Predictor import nrgnn_Predictor
from NoisyGL.predictor.RTGNN_Predictor import rtgnn_Predictor
from NoisyGL.predictor.CLNode_Predictor import clnode_Predictor
from NoisyGL.predictor.CGNN_Predictor import cgnn_Predictor
from NoisyGL.predictor.CRGNN_Predictor import crgnn_Predictor
from NoisyGL.predictor.PIGNN_Predictor import pignn_Predictor
from NoisyGL.predictor.RNCGLN_Predictor import rncgln_Predictor
from NoisyGL.predictor.R2LP_Predictor import r2lp_Predictor

import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Ignores only UserWarnings


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default='cora_ml',
                        choices=['cora_ml', 'wikics',  'products', 'children','history', 'photo',  'cornell', 'texas','washington', 'wisconsin'], 
                        help='Select dataset')
    parser.add_argument('--noise_type', type=str,
                        default='llm',
                        choices=['clean', 'uniform', 'pair', 'llm', 'topology', 'feature', 'confidence'], help='Type of label noise')
    parser.add_argument('--noise_rate', type=float,  default=None, help='Label noise rate, If set to None, the noise rate will be automatically derived from the LLM-based label noise in the dataset.')
    parser.add_argument('--method', type=str, default='smodel', choices=['lcat', 'smodel','forward', 'backward', 'coteaching', 'sce', 'jocor',  'apl',  'dgnn','cp',  'nrgnn', 'rtgnn','clnode',  'cgnn', 'crgnn',   'pignn','rncgln', 'r2lp'], help="Select methods")
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--data_root', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    return args


def run_single_exp(dataset, args):
    model_conf = load_conf(None, args.method, dataset.name)
    model_conf.model['n_feat'] = dataset.dim_feats
    model_conf.model['n_classes'] = 1 if dataset.n_classes <=2 else dataset.n_classes
    model_conf.training['debug'] = False
    
    predictor = eval(args.method + '_Predictor')(model_conf, dataset, args.device)
    results = predictor.train()
    return results


if __name__ == '__main__':

    args = load_args()
    data_conf = load_conf('./config/datasets/' + args.data + '.yaml')
    train_acc = []
    valid_acc = []
    test_acc = []
    for run, seed in enumerate(range(args.start_seed, args.runs + args.start_seed)):
        args.seed = seed
        setup_seed(args.seed)
        dataset = NoisyData(name=args.data, conf=data_conf, noise_type=args.noise_type, noise_rate=args.noise_rate, seed=args.seed, path=args.data_root, device=args.device)
        results = run_single_exp(dataset, args)
        
        train_acc.append(results['train']*100)
        valid_acc.append(results['valid']*100)
        test_acc.append(results['test']*100)
        print(f'Run {run+1}/{args.runs} finished: Train Acc: {results["train"]*100:.2f}, Valid Acc: {results["valid"]*100:.2f}, Test Acc: {results["test"]*100:.2f}')
    print(f'#{args.method}# Avg Train Acc: {np.mean(train_acc):.2f}, Avg Valid Acc: {np.mean(valid_acc):.2f}, Avg Test Acc: {np.mean(test_acc):.2f}')





