import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset


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

    def __init__(self, data_dir, set_idx=None, target_size=10, future_size=10, add_gradients=False,
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
        # sources[:, -target_size - future_size:-future_size, :] = 0

        sources = sequences[:, :-target_size - future_size]
        targets = sequences[:, -target_size - future_size:-future_size]
        futures = sequences[:, -future_size:]

        # Determine random training and test indices
        self.test_size = int(test_fraction * self.n_sequences)
        np.random.seed(seed)
        test_sample = np.random.choice(np.arange(self.n_sequences), self.test_size, replace=False)
        train_sample = np.delete(np.arange(self.n_sequences), test_sample, axis=0)

        # Select train or test set
        if test:
            self.source_data = sources[test_sample]
            self.target_data = targets[test_sample]
            self.future_data = futures[test_sample]
        else:
            self.source_data = sources[train_sample]
            self.target_data = targets[train_sample]
            self.future_data = futures[train_sample]

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
        self.w = nn.parameter.Parameter(torch.randn(1, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(out_features - 1))
        self.f = torch.sin
        self.device = device

    def forward(self, x):
        t = torch.FloatTensor(torch.arange(x.size(1), dtype=torch.float32)).unsqueeze(-1).to(self.device)
        v = t2v(t, self.f, self.w, self.b, self.w0, self.b0).repeat((x.size(0), 1, 1))
        return torch.cat((x, v), dim=-1)


class TimeDelay(nn.Module):
    """
    This module concatenates the features of previous timesteps to the features of each timestep as a form of embedding.

    dim: int, number of previous timesteps to be concatenated
    delay: int, how many timesteps are skipped between concatenated timesteps

    example:
    if dim=3 and delay=2
    [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]] -> [[3, 2, 1], [5, 4, 3], [7, 6, 5], [9, 8, 7]]
    """
    def __init__(self, dim, delay=1):
        super(TimeDelay, self).__init__()
        self.d = dim
        self.tau = delay

    def forward(self, x):
        new_sequence_length = (x.size(1) - self.d + 1) // self.tau
        x = x.flip(dims=[1])
        embeddings = [x[:, i:i + self.d * self.tau:self.tau].view(x.size(0), 1, -1) for i in range(new_sequence_length)]
        x = torch.cat(embeddings, dim=1).flip(dims=[1])
        return x


class FluxTransformer(nn.Module):
    """
    Transformer model for predicting the flux (and other features) of an AR.

    data_dir: directory where the data is stored
    set_idx: index of the dataset (1, 2, 3)
    target_size: size of target sequence we want to predict
    model_d: feature dimension of the transformer
    feature: which feature to use ('flux', 'area' or None, where None uses all features)
    output_feature: which feature the model should predict ('flux' or None, where None futures all source features)
    encoding: type of positional encoding ('pos' or 't2v')
    epochs: number of epochs
    batch_size: batch size
    learning_rate: learning rate
    device: device to use
    """

    def __init__(self, data_dir, set_id=None, source_size=100, target_size=25, future_size=25, model_d=32, nheads=1,
                 encoding='pos', time_encoding_dim=8, enc_layers=1, dec_layers=1, prediction_distance=1, epochs=100,
                 batch_size=64, learning_rate=1e-4, gamma=0.99,
                 device=torch.device("cpu")):
        super(FluxTransformer, self).__init__()
        self.epochs = epochs
        self.device = device
        self.prediction_distance = prediction_distance

        # Create datasets
        if set_id == 'sin':
            self.train_data = DataLoader(SinusoidDataset(n_sequences=8000, source_size=source_size,
                                                         target_size=target_size, future_size=future_size,
                                                         add_noise=True, device=self.device), batch_size=batch_size)
            self.test_data = DataLoader(SinusoidDataset(n_sequences=2000, source_size=source_size,
                                                        target_size=target_size, future_size=future_size,
                                                        add_noise=True, device=self.device), batch_size=batch_size)
        elif set_id == 'pol':
            self.train_data = DataLoader(PolynomialDataset(n_sequences=8000, source_size=source_size,
                                                           target_size=target_size, future_size=future_size,
                                                           add_noise=True, device=self.device), batch_size=batch_size)
            self.test_data = DataLoader(PolynomialDataset(n_sequences=2000, source_size=source_size,
                                                          target_size=target_size, future_size=future_size,
                                                          add_noise=True, device=self.device), batch_size=batch_size)
        else:
            self.train_data = DataLoader(FluxDataset(data_dir, set_idx=set_id, target_size=target_size,
                                                     future_size=future_size, test=False, seed=42,
                                                     device=self.device), batch_size=batch_size)
            self.test_data = DataLoader(FluxDataset(data_dir, set_idx=set_id, target_size=target_size,
                                                    future_size=future_size, test=True, seed=42,
                                                    device=self.device), batch_size=batch_size)

        # Store some dimensions of the data
        self.sequence_length = self.train_data.dataset.sequence_length
        self.target_size = target_size
        self.future_size = future_size
        self.source_size = self.sequence_length - self.target_size - self.future_size
        self.data_dim = self.train_data.dataset.n_features

        # Feature dimension the transformer will use
        self.model_dim = model_d

        # Positional encoding type
        self.encoding = encoding
        if encoding == 'pos':
            self.positional_encoding = nn.Sequential(nn.Linear(self.data_dim, self.model_dim),
                                                     PositionalEncoding(self.model_dim))
            self.delay_buffer = 0
        elif encoding == 't2v':
            self.positional_encoding = nn.Sequential(SineActivation(time_encoding_dim, self.device),
                                                     nn.Linear(time_encoding_dim + self.data_dim, self.model_dim),)
            self.delay_buffer = 0
        elif encoding == 'td':
            self.positional_encoding = nn.Sequential(TimeDelay(dim=time_encoding_dim, delay=1),
                                                     nn.Linear(time_encoding_dim * self.data_dim, self.model_dim))
            self.delay_buffer = time_encoding_dim

        # Mask for the target
        self.tgt_mask = self._generate_square_subsequent_mask(self.target_size).to(self.device)

        # Transformer model (pytorch's implementation of the transformer)
        self.transformer = nn.Transformer(d_model=self.model_dim, nhead=nheads, num_encoder_layers=enc_layers,
                                          num_decoder_layers=dec_layers, dropout=0.2, batch_first=True)

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
        src = seq[:, :self.source_size - self.delay_buffer]

        # (!) Use the points (target_size - 1) until the second to last point as the decoder input
        decoder_input = seq[:, -self.target_size - self.prediction_distance:-self.prediction_distance]

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
                print(f'Model not loaded. No model exists at {save_filename}')

        min_loss = np.inf

        n_train_batches = len(self.train_data)
        n_test_batches = len(self.test_data)

        for epoch in range(self.epochs):
            self.train()
            train_loss = []

            # Training
            for (src, tgt, _) in tqdm(self.train_data, desc=f'Epoch {epoch}', total=n_train_batches):
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
            for (src, tgt, _) in tqdm(self.test_data, desc=f'Epoch {epoch}', total=n_test_batches):
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

    def predict(self, src, tgt):
        with torch.no_grad():
            seq = torch.cat((src, tgt), dim=1)
            dummy_point = torch.zeros((seq.size(0), 1, seq.size(2))).to(self.device)
            for _ in range(self.future_size):
                # Add a dummy point with value 0 to the sequence (the value will be predicted by the model)
                seq = torch.cat((seq, dummy_point), dim=1)

                # Use the (target_size) last steps of the sequence as the target
                tgt = seq[:, -self.target_size:]

                # Use the (source_size) steps before the target as the source
                src = seq[:, -self.target_size - self.source_size:-self.target_size]

                # Calculate the target sequence
                out = self(src, tgt)

                # Use the last point of the calculated target sequence to replace the dummy point
                seq[:, -1] = out[:, -1]
        return seq[:, -self.future_size:]

    def show_example(self, src, tgt, ftr):
        self.eval()
        # Timesteps
        src_t = np.arange(self.source_size)
        tgt_t = np.arange(self.source_size, self.source_size + self.target_size)
        ftr_t = np.arange(self.source_size + self.target_size, self.sequence_length)

        # Prediction of the target
        pred_tgt = self(src, tgt)

        # Prediction further into the future
        pred_ftr = self.predict(src, tgt)

        fig, ax = plt.subplots(3, 3, figsize=(16, 12))

        for i in range(3):
            for j in range(3):
                ax[i, j].plot(src_t, src[3 * i + j, :, 0].cpu().detach().numpy(), label='source')
                ax[i, j].plot(tgt_t, tgt[3 * i + j, :, 0].cpu().detach().numpy(), label='target')
                ax[i, j].plot(tgt_t, pred_tgt[3 * i + j, :, 0].cpu().detach().numpy(), label='target_prediction')
                ax[i, j].plot(ftr_t, ftr[3 * i + j, :, 0].cpu().detach().numpy(), label='future')
                ax[i, j].plot(ftr_t, pred_ftr[3 * i + j, :, 0].cpu().detach().numpy(), label='future_prediction')
                ax[i, j].legend(fontsize=8)
                ax[i, j].set_xlabel('step')
                ax[i, j].set_ylabel('flux')
        plt.tight_layout()
        plt.savefig('transformer_results.png')
        plt.show()


if __name__ == "__main__":
    flux_data_dir = '/content/drive/MyDrive/adl/Magnetic_Flux_Area_Data'
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(dev)

    src_size = 150
    tgt_size = 25
    ftr_size = 100

    model_dim = 32
    nhead = 4
    t_encoding_dim = 8
    e_layers = 2
    d_layers = 2

    ep = 1

    set_id = 'pol'
    features = None
    train = True

    if set_id == 'sin':
        transformer_save_file = os.path.join(os.getcwd(), 'sinusoid_transformer_params')
        eval_data = DataLoader(SinusoidDataset(n_sequences=1000, source_size=src_size, target_size=tgt_size,
                                               future_size=ftr_size, add_noise=False, seed=14, device=dev),
                               batch_size=9)
    elif set_id == 'pol':
        transformer_save_file = os.path.join(os.getcwd(), 'polynomial_transformer_params')
        eval_data = DataLoader(PolynomialDataset(n_sequences=1000, source_size=src_size, target_size=tgt_size,
                                                 future_size=ftr_size, add_noise=True, seed=12, device=dev),
                               batch_size=9)
    else:
        transformer_save_file = os.path.join(os.getcwd(), 'flux_transformer_params')
        eval_data = DataLoader(FluxDataset(flux_data_dir, set_idx=set_id, target_size=tgt_size,
                                           future_size=ftr_size, test=True, seed=12, device=dev), batch_size=9)

    model = FluxTransformer(data_dir=flux_data_dir, set_id=set_id, source_size=src_size, target_size=tgt_size,
                            future_size=ftr_size, model_d=model_dim, nheads=nhead, encoding='td',
                            time_encoding_dim=t_encoding_dim, enc_layers=e_layers, dec_layers=d_layers,
                            prediction_distance=25, epochs=ep, learning_rate=5e-3, gamma=0.9, device=dev).to(dev)

    # Train or load the model
    if train:
        model.train_model(transformer_save_file, load_cp=False)
    else:
        model.load_model(transformer_save_file)

     # Plot some examples
    test_src, test_tgt, test_ftr = next(iter(eval_data))
    model.show_example(test_src, test_tgt, test_ftr)