import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

class SinusoidDataset(Dataset):
    """
    Dummy dataset that with sequences consisting of an added sine and cosine, which have different amplitudes and
    frequencies for each sequence.

    Parameters
    n_sequences: number of sequences in the dataset
    sequence_length: how many data point there are in each sequence
    target_size: how many data point does the target consist of, i.e. the data points we want to predict
    """
    def __init__(self, n_sequences=1000, sequence_length=125, target_size=25, add_noise=True, seed=42,
                 device=torch.device("cpu")):
        self.n_sequences = n_sequences
        self.sequence_length = sequence_length
        self.n_features = 1
        self.device = device

        self.target_size = target_size
        self.x = np.linspace(0, 1, sequence_length)

        # Determine the amplitudes and frequencies for each sequence
        np.random.seed(seed)
        self.amplitudes = np.random.uniform(low=-1.0, high=1.0, size=(n_sequences, 2))
        self.frequencies = np.random.uniform(high=np.pi * sequence_length / 8, size=(n_sequences, 2))

        # Optionally add noise
        self.add_noise = add_noise
        self.epsilon = np.random.normal(scale=0.05, size=(n_sequences, sequence_length))

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, item):
        a, b, = self.amplitudes[item]
        w_1, w_2 = self.frequencies[item]
        y = a * np.sin(w_1 * self.x) + b * np.cos(w_2 * self.x)
        if self.add_noise:
            y += self.epsilon[item]

        src = y[:-self.target_size][..., None]
        tgt = y[-self.target_size:][..., None]
        return torch.FloatTensor(src).to(self.device), torch.FloatTensor(tgt).to(self.device)
    
    
class CurveDataset(Dataset):
    """
    Dummy dataset that with sequences consisting of continuous random curves, with different coefficients and powers.
    
    curves are: y = a*(x+b)^c for each exponent c (goes from 0 + 5)

    Parameters
    n_sequences: number of sequences in the dataset
    sequence_length: how many data point there are in each sequence
    target_size: how many data point does the target consist of, i.e. the data points we want to predict
    """
    def __init__(self, n_sequences=1000, sequence_length=125, target_size=25, add_noise=True, seed=42, max_degree=5,
                 device=torch.device("cpu")):
        self.n_sequences = n_sequences
        self.sequence_length = sequence_length
        self.n_features = 1
        self.device = device

        self.target_size = target_size
        self.max_degree = max_degree
        self.x = np.linspace(0, sequence_length, sequence_length)

        # Determine the amplitudes and frequencies for each sequence
        np.random.seed(seed)
        self.degrees = np.random.choice([0, 1], size=(n_sequences, self.max_degree), p=[0.5, 0.5])
        self.coeff = np.random.uniform(low=-10.0, high=10.0, size=(n_sequences, self.max_degree))
        self.const = np.random.uniform(low=-self.sequence_length, high=self.sequence_length, size=(n_sequences, self.max_degree))
        self.zero_deg = np.random.choice([0, 1], size=(n_sequences, 1), p=[0.5, 0.5])
        self.q = np.random.uniform(low=-10.0, high=10.0, size=(n_sequences, 1))

        # Optionally add noise
        self.add_noise = add_noise
        self.epsilon = np.random.normal(scale=0.05, size=(n_sequences, sequence_length))

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, item):
        
        deg = self.degrees[item]
        angular = self.coeff[item]
        const = self.const[item]
        zero_deg = self.zero_deg[item]
        exp = [i for i in range(1, self.max_degree+1)]
        q = self.q[item]
        
        #d5, d4, d3, d2, d1, d0 = self.degrees[item]
        #m5, m4, m3, m2, m1 = self.coeff[item]
        #a5, a4, a3, a2, a1 = self.const[item]
        
        y = np.zeros(self.sequence_length)
        
        for d, m, c, e in zip(deg, angular, const, exp):
            tmp = d*m*(self.x + c)**e
            y = y + tmp
            
        y = y + zero_deg*q
        #if self.add_noise:
        #    y += self.epsilon[item]

        src = y[:-self.target_size][..., None]
        tgt = y[-self.target_size:][..., None]
        return torch.FloatTensor(src).to(self.device), torch.FloatTensor(tgt).to(self.device)


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
    def __init__(self, data_dir, set_idx=None, target_size=25, add_gradients=False,
                 test=False, test_fraction=0.33, seed=42, device=torch.device("cpu")):
        self.set_dir = f'set_{set_idx}'
        self.device = device

        # Load data.
        # Sources have shape [n_sequences, source_length, n_features]
        # Targets have shape [n_sequences, target_length, n_features]
        sources = np.load(os.path.join(data_dir, self.set_dir, 'input_res_concatenated.npy'))
        targets = np.load(os.path.join(data_dir, self.set_dir, 'target_res_concatenated.npy'))

        # Concatenate sources and targets for custom target size
        sequences = np.concatenate((sources, targets), axis=1)

        # Optionally add gradients of the features to the features.
        if add_gradients:
            sequences = np.concatenate((sequences, np.gradient(sequences, axis=1)), axis=-1)

        # Standardize the sequences
        sequences = normalize(sequences)

        # Store data shape
        self.n_sequences = sequences.shape[0]
        self.sequence_length = sequences.shape[1]
        self.n_features = sequences.shape[2]

        # The source for the model will be the sequence, but with its target data points masked.
        # sources = sequences.copy()
        # sources[:, -target_size:, :] = 0
        sources = sequences[:, :-target_size]

        # Targets are the 'target_size' final steps in the sequence.
        targets = sequences[:, -target_size:]

        # Determine random training and test indices
        self.test_size = int(test_fraction * self.n_sequences)
        np.random.seed(seed)
        test_sample = np.random.choice(np.arange(self.n_sequences), self.test_size, replace=False)
        train_sample = np.delete(np.arange(self.n_sequences), test_sample, axis=0)

        # Select train or test set
        if test:
            self.source_data = sources[test_sample]
            self.target_data = targets[test_sample]
        else:
            self.source_data = sources[train_sample]
            self.target_data = targets[train_sample]

    def __len__(self):
        return self.source_data.shape[0]

    def __getitem__(self, idx):
        src = self.source_data[idx]
        tgt = self.target_data[idx]
        return torch.FloatTensor(src).to(self.device), torch.FloatTensor(tgt).to(self.device)

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


def make_weighted_mse_loss(output_size, device):
    """
    Creates a mse loss functions that gives weights to the errors based on their position in the sequence. Predictions
    further in time are expected to have larger errors, so we apply smaller weights to later predictions to focus the
    learning on earlier timesteps.

    Parameters
    output_size : length of the sequence to be predicted
    device : device

    Returns
    A weighted mse loss function
    """
    weights = torch.arange(1, output_size + 1)
    weights = torch.flip(weights, dims=[0]).unsqueeze(0).unsqueeze(-1).to(device)

    def weighted_mse_loss(outputs, targets):
        return torch.mean(weights * (outputs[..., :1] - targets[..., :1]) ** 2)

    return weighted_mse_loss


class PositionalEncoding(nn.Module):
    """
    Positional encoding module. Adds a 'positional encoding' vector to the feature vector, which allows the model
    to derive the position of a datapoint from the features.

    d_model: number of features in the data
    max_len: maximum size of an source sequence
    dropout: dropout probability
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, src):
        # src has shape [batch_size, sequence_length, d_model]
        # self.pe has shape [1, max_len, d_model]
        src = src + self.pe[:, :src.size(1)]
        return self.dropout(src)


def t2v(tau, f, w, b, w0, b0, arg=None):
    """
    Time2Vec function. From timestep scalar tau, a vector that represents that timestep is created.
    """
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], -1)


class SineActivation(nn.Module):
    """
    A Time2Vec module with a sine as its periodic function (self.f). This is an alternative to positional encoding,
    which should work better for time series (instead of e.g. natural language problems).
    """
    def __init__(self, out_features, device):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(1, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin
        self.device = device

    def forward(self, x):
        t = torch.FloatTensor(torch.arange(x.size(1), dtype=torch.float32)).unsqueeze(-1).to(self.device)
        v = t2v(t, self.f, self.w, self.b, self.w0, self.b0).repeat((x.size(0), 1, 1))
        return torch.cat((x, v), dim=-1)


class FluxTransformer(nn.Module):
    """
    Transformer model for predicting the flux (and other features) of an AR.

    data_dir: directory where the data is stored
    set_idx: index of the dataset (1, 2, 3)
    target_size: size of target sequence we want to predict
    model_d: feature dimension of the transformer
    feature: which feature to use ('flux', 'area' or None, where None uses all features)
    output_feature: which feature the model should predict ('flux' or None, where None predicts all source features)
    encoding: type of positional encoding ('pos' or 't2v')
    epochs: number of epochs
    batch_size: batch size
    learning_rate: learning rate
    device: device to use
    """
    def __init__(self, data_dir, set_idx=None, target_size=25, model_d=32, nheads=1, encoding='pos', epochs=100,
                 batch_size=64, learning_rate=1e-4, gamma=0.99, device=torch.device("cpu")):
        super(FluxTransformer, self).__init__()
        self.epochs = epochs
        self.device = device

        # Create datasets
        if set_idx is None:
            self.train_data = DataLoader(SinusoidDataset(n_sequences=8000, sequence_length=125, target_size=target_size,
                                                         add_noise=False, device=self.device), batch_size=batch_size)
            self.test_data = DataLoader(SinusoidDataset(n_sequences=2000, sequence_length=125, target_size=target_size,
                                                        add_noise=False, device=self.device), batch_size=batch_size)
        else:
            self.train_data = DataLoader(FluxDataset(data_dir, set_idx=set_idx, target_size=target_size,
                                                     test=False, seed=42, device=self.device), batch_size=batch_size)
            self.test_data = DataLoader(FluxDataset(data_dir, set_idx=set_idx, target_size=target_size,
                                                    test=True, seed=42, device=self.device), batch_size=batch_size)

        # Store some dimensions of the data
        self.sequence_length = self.train_data.dataset.sequence_length
        self.target_size = target_size
        self.source_size = self.sequence_length - self.target_size
        self.data_dim = self.train_data.dataset.n_features

        # Feature dimension the transformer will use
        self.model_dim = model_d

        # Positional encoding type
        if encoding == 'pos':
            self.positional_encoding = nn.Sequential(nn.Linear(self.data_dim, self.model_dim),
                                                     PositionalEncoding(self.model_dim))
        elif encoding == 't2v':
            self.positional_encoding = SineActivation(self.model_dim - self.data_dim, self.device)

        # Mask for the target
        self.tgt_mask = self._generate_square_subsequent_mask(self.target_size).to(self.device)

        # Transformer model (pytorch's implementation of the transformer)
        self.transformer = nn.Transformer(d_model=self.model_dim, nhead=nheads, num_encoder_layers=1,
                                          num_decoder_layers=1, batch_first=True)

        # Previous model (need to change the src and tgt for this to work properly)
        # self.src_mask = self._generate_square_subsequent_mask(self.source_size).to(self.device)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=1, dropout=0.1,
        #                                                 dim_feedforward=2048, batch_first=True)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.decoder = nn.Linear(self.model_dim, self.data_dim)

        # Decoder initialization (I just copied this, don't know if it helps much)
        self.init_weights()

        # Loss, optimizer and learning rate scheduler
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.optimizer = optim.Adam([{'params': self.parameters()}], lr=learning_rate, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=gamma)
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=1e-4, max_lr=0.1, step_size_up=100)

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt):
        # Concatenate the source and target
        seq = torch.cat((src, tgt), dim=1)

        # Add positional encoding to each point in the sequence (this also changes the feature dimension to d_model)
        seq = self.positional_encoding(seq)

        # Use the (source_size) first steps as the source
        src = seq[:, :self.source_size]

        # (!) Use the points (target_size - 1) until the second to last point as the decoder input
        decoder_input = seq[:, -self.target_size - 1:-1]

        # Transformer model magic (out has the same shape as decoder_input)
        # The tgt_mask is the mask for decoder_input
        out = self.transformer(src, decoder_input, tgt_mask=self.tgt_mask)

        # Part of alternative model
        # out = self.transformer_encoder(src, self.src_mask)
        # out = out[:, -self.target_size:]

        # Change the feature dimension back to [flux, area]
        out = self.decoder(out)
        return out

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)
        return mask

    def train_model(self, save_filename=None, load_cp=False):
        if load_cp:
            if os.path.exists(save_filename):
                self.load_model(save_filename)
            else:
                print(f'Model not loaded. No model exits at {save_filename}')

        min_loss = np.inf

        n_train_batches = len(self.train_data)
        n_test_batches = len(self.test_data)

        for epoch in range(self.epochs):
            self.train()
            train_loss = []

            # Training
            for (src, tgt) in tqdm(self.train_data, desc=f'Epoch {epoch}', total=n_train_batches):
                self.optimizer.zero_grad()
                output = self(src, tgt)
                loss = self.loss_fn(output, tgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optimizer.step()
                train_loss.append(loss.item())

            mean_train_loss = np.mean(train_loss)
            print('Train loss:', mean_train_loss)

            self.eval()
            test_loss = []

            # Validation
            for (src, tgt) in tqdm(self.test_data, desc=f'Epoch {epoch}', total=n_test_batches):
                output = self(src, tgt)
                loss = self.loss_fn(output, tgt)
                test_loss.append(loss.item())

            mean_test_loss = np.mean(test_loss)
            print('Test loss:', mean_test_loss)

            # Check if the validation loss is smaller than it was for any previous epoch
            if mean_test_loss < min_loss:
                self.save_model(save_filename)
                min_loss = mean_test_loss

            # Adjust the learning rate
            # print(self.scheduler.get_last_lr())
            self.scheduler.step()

    def load_model(self, save_filename):
        self.load_state_dict(torch.load(save_filename))

    def save_model(self, save_filename):
        torch.save(self.state_dict(), save_filename)

    def predict(self, input_sequence, predict_size):
        with torch.no_grad():
            seq = input_sequence
            for _ in range(predict_size):
                # Add a dummy point with value 0 to the sequence (the value will be predicted by the model)
                
                new_tensor = torch.zeros((seq.size(0), 1, seq.size(2))).to(self.device)
                
                seq = torch.cat((seq, new_tensor), dim=1)

                # Use the (target_size) last steps of the sequence as the target
                tgt = seq[:, -self.target_size:]

                # Use the (source_size) steps before the target as the source
                src = seq[:, -self.target_size - self.source_size:-self.target_size]

                # Calculate the target sequence
                out = self(src, tgt)

                # Use the last point of the calculated target sequence to replace the dummy point
                seq[:, -1] = out[:, -1]
        return seq

    def show_example(self, src, tgt, predict_size):
        self.eval()
        # Prediction of the target
        pred_tgt = self(src, tgt)

        # Prediction further into the future
        pred = self.predict(torch.cat((src, tgt), dim=1), predict_size=predict_size)

        # Add first point of the target to the source for connected source and target in the plot
        src = torch.cat((src, tgt[:, :1]), dim=1)

        # Timesteps
        src_t = np.arange(src.size(1))
        tgt_t = np.arange(src.size(1) - 1, src.size(1) - 1 + self.target_size)
        pred_t = np.arange(pred.size(1))

        fig, ax = plt.subplots(3, 3, figsize=(16, 12))

        for i in range(3):
            for j in range(3):
                ax[i, j].plot(src_t, src[3 * i + j, :, 0].cpu().detach().numpy(), label='source')
                ax[i, j].plot(tgt_t, tgt[3 * i + j, :, 0].cpu().detach().numpy(), label='target')
                ax[i, j].plot(tgt_t, pred_tgt[3 * i + j, :, 0].cpu().detach().numpy(), label='train_prediction')
                ax[i, j].plot(pred_t[-predict_size - 1:], pred[3 * i + j, -predict_size - 1:, 0].cpu().detach().numpy(),
                              label='prediction')
                ax[i, j].legend(fontsize=8)
                ax[i, j].set_xlabel('step')
                ax[i, j].set_ylabel('flux')
        plt.tight_layout()
        plt.savefig('transformer_results.png')
        plt.show()