###### import modules ######
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


###### load dataset ######

def flux_train_test_eval(data_dir, set_idx, add_gradients=False,
                         test_fraction=0.33, seed=42, device=torch.device("cpu")):
    if set_idx == 4:
        class_call = NewFluxDataset
    else:
        class_call = FluxDataset
    
    train_dset = class_call(data_dir, set_idx, 
                            #target_size=TARGET_SIZE,
                            #future_size=ftr_size,
                            seed=seed,
                            device=device)
    test_dset = class_call(data_dir, set_idx, 
                           #target_size=TARGET_SIZE,
                           #future_size=ftr_size,
                           seed=seed,
                           test_set=True,
                           device=device)
    eval_dset = class_call(data_dir, set_idx, 
                           #target_size=TARGET_SIZE,
                           #future_size=ftr_size,
                           test_set=True, seed=seed+1,
                           device=device)
    return train_dset, test_dset, eval_dset

def example_train_test_eval(dset_type, 
                            n_sequences=1000, source_size=100, target_size=25, 
                            future_size=25, add_noise=True, seed=42,
                            test_fraction=0.33,
                            device=torch.device("cpu")):
    if dset_type == 'sin':
        class_call = SinusoidDataset
    elif dset_type == 'pol':
        class_call = PolynomialDataset
    
    test_nseq = int(test_fraction*n_sequences)
    train_dset = class_call(n_sequences=n_sequences, 
                            source_size=source_size, target_size=target_size, 
                            future_size=future_size, add_noise=add_noise, seed=seed,
                            device=device)
    test_dset = class_call(n_sequences=test_nseq, 
                           source_size=source_size, target_size=target_size, 
                           future_size=future_size, add_noise=add_noise, seed=seed,
                           device=device)
    eval_dset = class_call(n_sequences=test_nseq, 
                           source_size=source_size, target_size=target_size, 
                           future_size=future_size, add_noise=add_noise, seed=seed+1,
                           device=device)
    return train_dset, test_dset, eval_dset


###### normalize data ######

def normalize(sequences):
    """
    Standardize sequences by subtracting the mean and dividing by the standard deviation.

    Parameters
    sequences : Sequence tensor with shape [n_sequences, sequence_length, n_features]

    Returns
    Normalized sequences
    """
    mean = np.mean(sequences, axis=(0, 1))
    std = np.std(sequences, axis=(0, 1))

    normalized_sequences = (sequences - mean) / std
    return normalized_sequences


###### dummy dataset classes ######

class SinusoidDataset(Dataset):
    """
    Dummy dataset that with sequences consisting of an added sine and cosine, which have different amplitudes and
    frequencies for each sequence.

    Parameters
    n_sequences: number of sequences in the dataset
    sequence_length: how many data point there are in each sequence
    target_size: how many data point does the target consist of, i.e. the data points we want to predict
    """
    def __init__(self, n_sequences=1000, source_size=100, target_size=25, future_size=25, add_noise=True, seed=42,
                 device=torch.device("cpu")):
        self.n_sequences = n_sequences
        self.n_features = 1
        self.device = device

        self.source_size = source_size
        self.target_size = target_size
        self.future_size = future_size
        self.sequence_length = source_size + target_size + future_size

        self.x = np.linspace(0, 1, self.sequence_length)

        # Determine the amplitudes and frequencies for each sequence
        np.random.seed(seed)
        self.amplitudes = np.random.uniform(low=0.5, high=1.0, size=(n_sequences, 2))
        self.frequencies = np.random.uniform(low=8 * np.pi, high=32 * np.pi, size=(n_sequences, 2))

        # Optionally add noise
        self.add_noise = add_noise
        self.epsilon = np.random.normal(scale=0.05, size=(n_sequences, self.sequence_length))
        
    def get_source_size(self):
        return self.source_size
    
    def get_target_size(self):
        return self.target_size
    
    def get_future_size(self):
        return self.future_size
    
    def get_sequence_length(self):
        return self.sequence_length
    
    def get_n_features(self):
        return self.n_features

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, item):
        a, b, = self.amplitudes[item]
        w_1, w_2 = self.frequencies[item]
        y = a * np.sin(w_1 * self.x)  # + b * np.cos(w_2 * self.x)
        if self.add_noise:
            y += self.epsilon[item]

        src = y[:self.source_size][..., None]
        tgt = y[self.source_size:-self.future_size][..., None]
        ftr = y[-self.future_size:][..., None]

        src = torch.FloatTensor(src).to(self.device)
        tgt = torch.FloatTensor(tgt).to(self.device)
        ftr = torch.FloatTensor(ftr).to(self.device)
        return src, tgt, ftr

class PolynomialDataset(Dataset):
    """
    Dummy dataset that with sequences consisting of an added sine and cosine, which have different amplitudes and
    frequencies for each sequence.

    Parameters
    n_sequences: number of sequences in the dataset
    sequence_length: how many data point there are in each sequence
    target_size: how many data point does the target consist of, i.e. the data points we want to predict
    """

    def __init__(self, n_sequences=1000, source_size=100, target_size=25, future_size=25, add_noise=True, seed=42,
                 device=torch.device("cpu")):
        self.n_sequences = n_sequences
        self.n_features = 1
        self.device = device

        self.source_size = source_size
        self.target_size = target_size
        self.future_size = future_size
        self.sequence_length = source_size + target_size + future_size

        self.x = np.linspace(0, 1, self.sequence_length)

        # Determine the amplitudes and frequencies for each sequence
        np.random.seed(seed)
        self.exponents = np.random.choice([1, 2, 3], size=self.n_sequences)
        self.weight = np.random.uniform(low=-1.0, high=1.0, size=self.n_sequences)
        self.bias = np.random.uniform(low=-1.0, high=1.0, size=self.n_sequences)

        # Optionally add noise
        self.add_noise = add_noise
        self.epsilon = np.random.normal(scale=0.002, size=(n_sequences, self.sequence_length))
        
    def get_source_size(self):
        return self.source_size
    
    def get_target_size(self):
        return self.target_size
    
    def get_future_size(self):
        return self.future_size
    
    def get_sequence_length(self):
        return self.sequence_length
    
    def get_n_features(self):
        return self.n_features

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, item):
        exp = self.exponents[item]
        a = self.weight[item]
        b = self.bias[item]
        y = a * self.x ** exp + b
        if self.add_noise:
            y += self.epsilon[item]

        src = y[:self.source_size][..., None]
        tgt = y[self.source_size:-self.future_size][..., None]
        ftr = y[-self.future_size:][..., None]

        src = torch.FloatTensor(src).to(self.device)
        tgt = torch.FloatTensor(tgt).to(self.device)
        ftr = torch.FloatTensor(ftr).to(self.device)
        return src, tgt, ftr
    
    
###### flux dataset classes ######
    
class NewFluxDataset(Dataset):
    """
    Dataset containing sequences of the fluxes (and areas) of AR's.
    data_dir: directory where the data is stored
    set_idx: index of the dataset (1, 2, 3)
    target_size: size of target sequence we want to predict
    feature: which feature to use ('flux', 'area' or None, where None uses all features)
    add_gradients: whether to add gradients of the features to the features
    test: whether to return a test dataset
    test_fraction: fraction of the data to use for the test data
    seed: random seed for selecting test/train data
    """

    def __init__(self, data_dir, 
                 set_idx=None, target_size=25, future_size=25, add_gradients=False,
                 test_set=False, test_fraction=0.33, seed=42, device=torch.device("cpu")):
        self.set_dir = f'set_{set_idx}'
        self.device = device
        
        # Load data.
        if test_set:
            data_file = os.path.join(data_dir, self.set_dir, 'test_normalized.npy')
        else:
            data_file = os.path.join(data_dir, self.set_dir, 'train_normalized.npy')
            
        sequences = np.load(data_file, allow_pickle=True)

        # Optionally add gradients of the features to the features.
        if add_gradients:
            sequences = np.concatenate((sequences, np.gradient(sequences, axis=1)), axis=-1)

        # Store data shape
        self.n_sequences = sequences.shape[0]
        self.sequence_length = sequences.shape[1]
        self.n_features = sequences.shape[2]
        self.target_size = target_size
        self.future_size = future_size

        self.source_data = sequences[:, :-target_size - future_size]
        self.target_data = sequences[:, -target_size - future_size:-future_size]
        self.future_data = sequences[:, -future_size:]
    
    def get_target_size(self):
        return self.target_size
    
    def get_future_size(self):
        return self.future_size
    
    def get_sequence_length(self):
        return self.sequence_length
    
    def get_n_features(self):
        return self.n_features

    def __len__(self):
        return self.source_data.shape[0]

    def __getitem__(self, idx):
        src = self.source_data[idx]
        tgt = self.target_data[idx]
        ftr = self.future_data[idx]

        src = torch.FloatTensor(src).to(self.device)
        tgt = torch.FloatTensor(tgt).to(self.device)
        ftr = torch.FloatTensor(ftr).to(self.device)
        return src, tgt, ftr

class FluxDataset(Dataset):
    """
    Dataset containing sequences of the fluxes (and areas) of AR's.

    data_dir: directory where the data is stored
    set_idx: index of the dataset (1, 2, 3)
    target_size: size of target sequence we want to predict
    feature: which feature to use ('flux', 'area' or None, where None uses all features)
    add_gradients: whether to add gradients of the features to the features
    test: whether to return a test dataset
    test_fraction: fraction of the data to use for the test data
    seed: random seed for selecting test/train data
    """

    def __init__(self, data_dir, set_idx=None, 
                 #target_size=10, 
                 #future_size=10, 
                 add_gradients=False,
                 test_set=False, test_fraction=0.33, seed=42, device=torch.device("cpu")):
        self.set_dir = f'set_{set_idx}'
        self.device = device

        # Load data.
        # Sources have shape [n_sequences, source_length, n_features]
        # Targets have shape [n_sequences, target_length, n_features]
        sources = np.load(os.path.join(data_dir, self.set_dir, 'input_res_concatenated.npy'))
        targets = np.load(os.path.join(data_dir, self.set_dir, 'target_res_concatenated.npy'))
        
        self.source_size = len(sources[0])
        self.target_size = len(targets[0])
        self.future_size = self.target_size

        # Concatenate sources and targets for custom target size
        sequences = np.concatenate((sources, targets), axis=1)

        # Optionally add gradients of the features to the features.
        if add_gradients:
            sequences = np.concatenate((sequences, np.gradient(sequences, axis=1)), axis=-1)

        # Standardize the sequences
        sequences = normalize(sequences)

        # Store data shape
        self.n_sequences = sequences.shape[0]
        self.sequence_length = sequences.shape[1] #this is source_size + target_size
        self.n_features = sequences.shape[2]

        # The source for the model will be the sequence, but with its target data points masked.
        # sources = sequences.copy()
        # sources[:, -target_size - future_size:-future_size, :] = 0

        sources = sequences[:, :-self.target_size - self.future_size]
        targets = sequences[:, -self.target_size - self.future_size:-self.future_size]
        futures = sequences[:, -self.future_size:]

        # Determine random training and test indices
        self.test_size = int(test_fraction * self.n_sequences)
        np.random.seed(seed)
        test_sample = np.random.choice(np.arange(self.n_sequences), self.test_size, replace=False)
        train_sample = np.delete(np.arange(self.n_sequences), test_sample, axis=0)

        # Select train or test set
        if test_set:
            self.source_data = sources[test_sample]
            self.target_data = targets[test_sample]
            self.future_data = futures[test_sample]
        else:
            self.source_data = sources[train_sample]
            self.target_data = targets[train_sample]
            self.future_data = futures[train_sample]
            
    def get_source_size(self):
        return self.source_size
    
    def get_target_size(self):
        return self.target_size
    
    def get_future_size(self):
        return self.future_size
    
    def get_sequence_length(self):
        return self.sequence_length
    
    def get_n_features(self):
        return self.n_features

    def __len__(self):
        return self.source_data.shape[0]

    def __getitem__(self, idx):
        src = self.source_data[idx]
        tgt = self.target_data[idx]
        ftr = self.future_data[idx]

        src = torch.FloatTensor(src).to(self.device)
        tgt = torch.FloatTensor(tgt).to(self.device)
        ftr = torch.FloatTensor(ftr).to(self.device)
        return src, tgt, ftr