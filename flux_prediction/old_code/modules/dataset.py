###### import modules ######
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


###### load dataset ######

def mlabel_train_test_eval(data_dir, n_labels=5, test_fraction=0.66, seed=42, device=torch.device("cpu")):
    train_dset = MultiLabelDataset(data_dir,
                                   n_labels=n_labels,
                                   test_fraction=test_fraction,
                                   seed=seed,
                                   device=device)
    test_dset = MultiLabelDataset(data_dir,
                                  n_labels=n_labels,
                                  test_fraction=test_fraction,
                                  seed=seed,
                                  test_set=True,
                                  device=device)
    eval_dset = MultiLabelDataset(data_dir,
                                  n_labels=n_labels,
                                  test_fraction=test_fraction,
                                  val_set=True, 
                                  seed=seed+1,
                                  device=device)
    return train_dset, test_dset, eval_dset

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


###### multilabel dataset ######

class MultiLabelDataset(Dataset):
    def __init__(self, data_dir, n_labels=5,
                 test_set=False, val_set=False, test_fraction=0.66, seed=42, device=torch.device("cpu")):
        self.device = device

        # Load data.
        # Sources have shape [n_sequences, source_length, n_features]
        # Targets have shape [n_sequences, target_length, n_features]
        
        if test_set:
            sources = np.load(os.path.join(data_dir, 'test_input.npy'))
            targets = np.load(os.path.join(data_dir, 'test_target.npy'))
        elif val_set:
            """
            sources = np.load(os.path.join(data_dir, 'val_input.npy'))
            targets = np.load(os.path.join(data_dir, 'val_target.npy'))
            """
            sources = np.load(os.path.join(data_dir, 'test_input.npy'))
            targets = np.load(os.path.join(data_dir, 'test_target.npy'))
        else:
            sources = np.load(os.path.join(data_dir, 'train_input.npy'))
            targets = np.load(os.path.join(data_dir, 'train_target.npy'))

        # Store data shape
        self.source_size = len(sources[0])
        self.target_size = len(targets[0])
        self.n_labels = n_labels
        self.str_n_sequences = len(sources)
        self.sequence_length = self.source_size + self.target_size #this is source_size + target_size
        self.n_features = len(sources[0][0]) #+ len(targets[0][0])

        # Determine random training and test indices
        if n_labels < 5:
            targets = self.relabel_data(targets)
        self.minvalue, self.maxvalue = self.min_max_value(sources)   
        # Determine random training and test indices
        if not val_set:
            self.test_size = int(test_fraction * self.str_n_sequences)
            np.random.seed(seed)
            test_sample = np.random.choice(np.arange(self.str_n_sequences), self.test_size, replace=False)
            train_sample = np.delete(np.arange(self.str_n_sequences), test_sample, axis=0)
            self.source_data = sources[test_sample]
            self.target_data = targets[test_sample]
        else:
            self.source_data = sources
            self.target_data = targets
        
        
        #if not test_set and not val_set:
        #self.source_data = self.apply_mask(sources[test_sample])
        #else:
        
        
        self.n_sequences = len(self.source_data)
        
    def relabel_data(self, targets):
        new_targets = np.zeros((self.str_n_sequences, self.target_size, self.n_labels))
        for i, tgt in enumerate(targets):
            old_label_idx = np.argmax(tgt[0])
            if old_label_idx > 0:
                if self.n_labels == 2:
                    new_label_idx = 1
                elif self.n_labels == 3:
                    if old_label_idx < 3:
                        new_label_idx = 1
                    else:
                        new_label_idx = 2
                new_targets[i,0,new_label_idx] = 1.
        return new_targets
          
    def pad_data(self, sources, targets):
        out_sources = np.pad(sources, ((0,0), (0,0), (0,self.n_labels)), mode='constant')
        out_targets = np.pad(targets, ((0,0), (0,0), (self.unpad_n_features,0)), mode='constant')
        return out_sources, out_targets
    
    def min_max_value(self, sources):
        flat = np.ravel(sources)
        minvalue = np.amin(flat)
        maxvalue = np.amax(flat)
        return minvalue, maxvalue
    
    def apply_mask(self, sources, masked=0.15, masked_ratios=[0.8, 0.1, 0.1]):
        """
        masks the sequence values, masked values are 15%
        of these values 80% are set to -inf, 10% to a random value and 10% to the actual value
        """
        n_elements = self.source_size**2
        n_idx = int(masked*n_elements)
        
        print(sources[0][0])
        
        for batch in tqdm(sources, desc='Masking sources', total=len(sources)):
            for timestep in batch:
                idx_l = [i for i in np.random.permutation(self.n_features)]
                idx_l = idx_l[:n_idx]

                for idx in idx_l:
                    if torch.rand(1) < 0.8:
                        timestep[idx] = float('-inf')
                    else:
                        if torch.rand(1) < 0.5:
                            timestep[idx] = (self.maxvalue - self.minvalue) * np.random.random_sample() + self.minvalue
                        else:
                            pass
        
        return sources
    
    def get_minvalue(self):
        return self.minvalue
    
    def get_maxvalue(self):
        return self.maxvalue
        
    def get_source_size(self):
        return self.source_size
    
    def get_target_size(self):
        return self.target_size
    
    def get_future_size(self):
        return None
    
    def get_sequence_length(self):
        return self.sequence_length
    
    def get_n_features(self):
        return self.n_features
                             
    def get_n_labels(self):
        return self.n_labels

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        src = self.source_data[idx]
        tgt = self.target_data[idx]

        src = torch.FloatTensor(src).to(self.device)
        tgt = torch.FloatTensor(tgt).to(self.device)
        return src, tgt
        

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