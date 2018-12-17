import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import pandas as pd
import numpy as np
from sklearn import preprocessing
import argparse
from Data.data_pipeline import *

def Parser(argparse):

        parser = argparse.ArgumentParser(description='VAE MNIST Example')
        parser.add_argument('--hidden-layers', type=list, default=[200,100], metavar='N',
                    help='how many layers will the encoder have')
        parser.add_argument('--encoding-dim', type=int, default=10, metavar='N',
                    help='encoding dimension')
        parser.add_argument('--learning-rate', type=int, default=0.001, metavar='N',
                    help='how many layers will the encoder have')
        parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 128)')
        parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
        parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
        args = parser.parse_args(args=[])


        args.cuda = not args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(args.seed)

        device = torch.device("cuda" if args.cuda else "cpu")

        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        
        return kwargs, device, args
    

    

def scale_data(data):
    return preprocessing.MinMaxScaler().fit_transform(data)

class ReadDataset_train_random(Dataset):
   #""" Diabetes dataset."""

   # Initialize your data, download, etc.
    def __init__(self):
        x = np.loadtxt('random.csv',
                       delimiter=',', dtype=np.float32)
        y = np.loadtxt('random.csv',
                       delimiter=',', dtype=np.float32)        
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

class ReadDataset_test_random(Dataset):
   #""" Diabetes dataset."""

   # Initialize your data, download, etc.
    def __init__(self):
        x = np.loadtxt('random_test.csv',
                       delimiter=',', dtype=np.float32)
        y = np.loadtxt('random_test.csv',
                       delimiter=',', dtype=np.float32)        
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len  

class ReadDataset_train_E_coli(Dataset):
   #""" Diabetes dataset."""

   # Initialize your data, download, etc.
    def __init__(self):
        root_gene = None
        minimum_evidence = 'weak'
        max_depth = np.inf
        r_expr, gene_symbols, sample_names = load_data(root_gene=root_gene,
                                     minimum_evidence=minimum_evidence,
                                     max_depth=max_depth)

        # Split data into train and test sets
        train_idxs, test_idxs = split_train_test(sample_names)
        expr_train = r_expr[train_idxs, :]
        expr_test = r_expr[test_idxs, :]
        
        x = expr_train.astype(np.float32)
        y = expr_train.astype(np.float32) 
        x_n=normalize(x)
        y_n=normalize(y)
        x=scale_data(x_n)
        y=scale_data(y_n)
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

class ReadDataset_test_E_coli(Dataset):
   #""" Diabetes dataset."""

   # Initialize your data, download, etc.
    def __init__(self):
        root_gene = None
        minimum_evidence = 'weak'
        max_depth = np.inf
        r_expr, gene_symbols, sample_names = load_data(root_gene=root_gene,
                                     minimum_evidence=minimum_evidence,
                                     max_depth=max_depth)

        # Split data into train and test sets
        train_idxs, test_idxs = split_train_test(sample_names)
        expr_train = r_expr[train_idxs, :]
        expr_test = r_expr[test_idxs, :]
        
        x = expr_test.astype(np.float32)
        y = expr_test.astype(np.float32)
        x_n=normalize(x)
        y_n=normalize(y)
        x=scale_data(x_n)
        y=scale_data(y_n)        
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len  
    
class ReadDataset_train_E_coli_genes(Dataset):
   #""" Diabetes dataset."""

   # Initialize your data, download, etc.
    def __init__(self):
        root_gene = None
        minimum_evidence = 'weak'
        max_depth = np.inf
        r_expr, gene_symbols, sample_names = load_data(root_gene=root_gene,
                                     minimum_evidence=minimum_evidence,
                                     max_depth=max_depth)
        r_expr=r_expr.T

        # Split data into train and test sets
#        train_idxs, test_idxs = split_train_test(gene_symbols)
        expr_train = r_expr[:int(r_expr.shape[0]*0.8), :]
        expr_test = r_expr[int(r_expr.shape[0]*0.8):, :]
        
        x = expr_train.astype(np.float32)
        y = expr_train.astype(np.float32) 
        x_n=normalize(x)
        y_n=normalize(y)
        x=scale_data(x_n)
        y=scale_data(y_n)
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

class ReadDataset_test_E_coli_genes(Dataset):
   #""" Diabetes dataset."""

   # Initialize your data, download, etc.
    def __init__(self):
        root_gene = None
        minimum_evidence = 'weak'
        max_depth = np.inf
        r_expr, gene_symbols, sample_names = load_data(root_gene=root_gene,
                                     minimum_evidence=minimum_evidence,
                                     max_depth=max_depth)
        r_expr=r_expr.T

        # Split data into train and test sets
        #train_idxs, test_idxs = split_train_test(gene_symbols)
        expr_train = r_expr[:int(r_expr.shape[0]*0.8), :]
        expr_test = r_expr[int(r_expr.shape[0]*0.8):, :]
        
        x = expr_test.astype(np.float32)
        y = expr_test.astype(np.float32)
        x_n=normalize(x)
        y_n=normalize(y)
        x=scale_data(x_n)
        y=scale_data(y_n)
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len  
        

def normalize(expr, kappa=1):
    """
    Normalizes expressions to make each gene have mean 0 and std kappa^-1
    :param expr: matrix of gene expressions. Shape=(nb_samples, nb_genes)
    :param kappa: kappa^-1 is the gene std
    :return: normalized expressions
    """
    mean = np.mean(expr, axis=0)
    std = np.std(expr, axis=0)
    return (expr - mean) / (kappa * std)


def restore_scale(expr, mean, std):
    """
    Makes each gene j have mean_j and std_j
    :param expr: matrix of gene expressions. Shape=(nb_samples, nb_genes)
    :param mean: vector of gene means. Shape=(nb_genes,)
    :param std: vector of gene stds. Shape=(nb_genes,)
    :return: Rescaled gene expressions
    """
    return expr * std + mean


def clip_outliers(expr, r_min, r_max):
    """
    Clips expression values to make them be between r_min and r_max
    :param expr: matrix of gene expressions. Shape=(nb_samples, nb_genes)
    :param r_min: minimum expression value (float)
    :param r_max: maximum expression value (float)
    :return: Clipped expression matrix
    """
    expr_c = np.copy(expr)
    expr_c[expr_c < r_min] = r_min
    expr_c[expr_c > r_max] = r_max
    return expr_c
    