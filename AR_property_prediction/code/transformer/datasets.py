import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SinusoidDataset(Dataset):
    """
    Dummy dataset that with sequences consisting of an added sine and cosine, which have different amplitudes and
    frequencies for each sequence.
    Parameters
    n_sequences: number of sequences in the dataset
    source_size: how many data points are in the source
    target_size: how many data points are in the target
    future_size: how many data points are in the future
    add_noise: whether to add a bit of noise to the data
    seed: the seed determines the random amplitudes and frequencies
    device: on what device to store the data
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

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, item):
        a, b, = self.amplitudes[item]
        w_1, w_2 = self.frequencies[item]
        sequences = a * np.sin(w_1 * self.x)  # + b * np.cos(w_2 * self.x)
        if self.add_noise:
            sequences += self.epsilon[item]

        src = sequences[:self.source_size][..., None]
        tgt = sequences[self.source_size:self.source_size + self.target_size][..., None]
        ftr = sequences[self.source_size + self.target_size:self.source_size + self.target_size + self.future_size][..., None]

        src = torch.FloatTensor(src).to(self.device)
        tgt = torch.FloatTensor(tgt).to(self.device)
        ftr = torch.FloatTensor(ftr).to(self.device)
        return src, tgt, ftr


class PolynomialDataset(Dataset):
    """
    Dummy dataset with sequences consisting of polynomials up to third order.
    Parameters
    n_sequences: number of sequences in the dataset
    source_size: how many data points are in the source
    target_size: how many data points are in the target
    future_size: how many data points are in the future
    add_noise: whether to add a bit of noise to the data
    seed: the seed determines the random amplitudes and frequencies
    device: on what device to store the data
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

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, item):
        exp = self.exponents[item]
        a = self.weight[item]
        b = self.bias[item]
        sequences = a * self.x ** exp + b
        if self.add_noise:
            sequences += self.epsilon[item]

        src = sequences[:self.source_size][..., None]
        tgt = sequences[self.source_size:self.source_size + self.target_size][..., None]
        ftr = sequences[self.source_size + self.target_size:self.source_size + self.target_size + self.future_size][..., None]

        src = torch.FloatTensor(src).to(self.device)
        tgt = torch.FloatTensor(tgt).to(self.device)
        ftr = torch.FloatTensor(ftr).to(self.device)
        return src, tgt, ftr


class FluxDataset(Dataset):
    """
    Dataset containing sequences of the fluxes (and areas) of AR's.
    data_dir: directory where the data is stored
    set_idx: identifier of the dataset
    source_size: size of source sequence
    target_size: size of target sequence we want to predict
    future_size: size of future sequence we want to predict
    add_gradients: whether to add gradients of the features to the features
    validation: whether to return a validation dataset (if False, returns train dataset)
    device: on what device to store the data
    """

    def __init__(self, data_dir, set_idx=None, source_size=40, target_size=10, future_size=10, add_gradients=False,
                 validation=False, device=torch.device("cpu")):
        self.set_dir = f'set_{set_idx}'
        self.device = device

        selected_keys = np.array([0, 1, 2, 3, 4, 7, 9, 10, 11, 12])

        if validation:
            data_file = os.path.join(data_dir, self.set_dir, 'validation_normalized.npy')
        else:
            data_file = os.path.join(data_dir, self.set_dir, 'train_normalized.npy')
        sequences = np.load(data_file, allow_pickle=True)[:, :, selected_keys]

        # Optionally add gradients of the features to the features.
        if add_gradients:
            sequences = np.concatenate((sequences, np.gradient(sequences, axis=1)), axis=-1)

        # Store data shape
        self.n_sequences = sequences.shape[0]
        self.sequence_length = sequences.shape[1]
        self.n_features = sequences.shape[2]

        self.source_data = sequences[:, :source_size]
        self.target_data = sequences[:, source_size:source_size + target_size]
        self.future_data = sequences[:, source_size + target_size:source_size + target_size + future_size]

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
