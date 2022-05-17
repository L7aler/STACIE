###### import modules ######

import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np


###### define loss functions ######

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


###### encoding classes ######

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
    
    
###### transformer module ######

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

    def __init__(self, train_dset, test_dset, multilabel=False,
                 model_d=32, nheads=1, encoding='pos', time_encoding_dim=8, 
                 enc_layers=1, dec_layers=1, prediction_distance=1, epochs=100,
                 batch_size=64, learning_rate=1e-4, gamma=0.99, device=torch.device("cpu")):
        
        super(FluxTransformer, self).__init__()
        self.multilabel = multilabel
        self.epochs = epochs
        self.device = device
        self.prediction_distance = prediction_distance

        #Load datasets
        self.train_data = DataLoader(train_dset, batch_size=batch_size)
        self.test_data = DataLoader(test_dset, batch_size=batch_size)

        # Store some dimensions of the data
        self.sequence_length = train_dset.get_sequence_length()
        self.target_size = train_dset.get_target_size()
        self.data_in_dim = train_dset.get_n_features()
        if not self.multilabel:
            self.data_out_dim = self.data_in_dim
            self.future_size = train_dset.get_future_size()
            self.source_size = self.sequence_length - self.target_size - self.future_size #?????
        else:
            self.source_size = 10
            self.data_out_dim = 45

        # Feature dimension the transformer will use
        self.model_dim = model_d
        
        #set time encoding dim
        self.time_encoding_dim = time_encoding_dim

        # Positional encoding type
        self.encoding = encoding
        self.positional_encoding, self.delay_buffer = self.set_encoding(encoding, data_dim=self.data_in_dim, model_dim=self.model_dim,
                                                                        time_encoding_dim=self.time_encoding_dim, device=self.device)
        
        # Mask for the target
        self.tgt_mask = self._generate_square_subsequent_mask(self.target_size).to(self.device)

        # Transformer model (pytorch's implementation of the transformer)
        self.transformer = nn.Transformer(d_model=self.model_dim, nhead=nheads, num_encoder_layers=enc_layers,
                                          num_decoder_layers=dec_layers, dropout=0.2, batch_first=True)

        self.decoder = nn.Linear(self.model_dim, self.data_out_dim)
        #change to the number of lables to apply to Louis' data

        # Decoder initialization (I just copied this, don't know if it helps much)
        self.init_weights()

        # Loss, optimizer and learning rate scheduler
        if not self.multilabel:
            self.loss_fn = nn.MSELoss(reduction='mean')
        else:
            self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = optim.Adam([{'params': self.parameters()}], lr=learning_rate, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=gamma)
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=1e-4, max_lr=0.1, step_size_up=100)
        
        #loss per epoch
        self.mean_train_loss = []
        self.mean_test_loss = []
        
    def set_encoding(self, encoding, data_dim=None, model_dim=None, time_encoding_dim=None, device=None):
        if encoding == 'pos':
            positional_encoding = nn.Sequential(nn.Linear(data_dim, model_dim), PositionalEncoding(model_dim))
            delay_buffer = 0
        elif encoding == 't2v':
            positional_encoding = nn.Sequential(SineActivation(time_encoding_dim, device), 
                                                nn.Linear(time_encoding_dim + data_dim, model_dim))
            delay_buffer = 0
        elif encoding == 'td':
            positional_encoding = nn.Sequential(TimeDelay(dim=time_encoding_dim, delay=1),
                                                nn.Linear(time_encoding_dim * data_dim, model_dim))
            delay_buffer = time_encoding_dim
            
        return positional_encoding, delay_buffer

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt):
        # Concatenate the source and target
        #print(src.shape, tgt.shape)
        seq = torch.cat((src, tgt), dim=1)
        #print(seq.shape)

        # Add positional encoding to each point in the sequence (this also changes the feature dimension to d_model)
        seq = self.positional_encoding(seq)

        # Use the (source_size) first steps as the source
        src = seq[:, :self.source_size - self.delay_buffer]

        # (!) Use the points (target_size - 1) until the second to last point as the decoder input
        decoder_input = seq[:, -self.target_size - self.prediction_distance :-self.prediction_distance]

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
            
            if self.multilabel:
                for (src, tgt) in tqdm(self.train_data, desc=f'Epoch {epoch}', total=n_train_batches):
                    self.optimizer.zero_grad()
                    output = self(src, tgt)
                    loss = self.loss_fn(output, tgt) #change to 
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                    self.optimizer.step()
                    train_loss.append(loss.item())
            else:
                for (src, tgt, _) in tqdm(self.train_data, desc=f'Epoch {epoch}', total=n_train_batches):
                    self.optimizer.zero_grad()
                    output = self(src, tgt)
                    loss = self.loss_fn(output, tgt) #change to 
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                    self.optimizer.step()
                    train_loss.append(loss.item())
            

            current_mean_train_loss = np.mean(train_loss)
            self.mean_train_loss.append(current_mean_train_loss)
            print('Train loss:', current_mean_train_loss)

            self.eval()
            test_loss = []

            # Validation
            if self.multilabel:
                for (src, tgt) in tqdm(self.test_data, desc=f'Epoch {epoch}', total=n_test_batches):
                    output = self(src, tgt)
                    loss = self.loss_fn(output, tgt)
                    test_loss.append(loss.item())
            else:
                for (src, tgt, _) in tqdm(self.test_data, desc=f'Epoch {epoch}', total=n_test_batches):
                    output = self(src, tgt)
                    loss = self.loss_fn(output, tgt)
                    test_loss.append(loss.item())

            current_mean_test_loss = np.mean(test_loss)
            self.mean_test_loss.append(current_mean_test_loss)
            print('Test loss:', current_mean_test_loss)

            # Check if the validation loss is smaller than it was for any previous epoch
            if current_mean_test_loss < min_loss:
                self.save_model(save_filename)
                min_loss = current_mean_test_loss

            # Adjust the learning rate
            # print(self.scheduler.get_last_lr())
            self.scheduler.step()

    def load_model(self, save_filename):
        self.load_state_dict(torch.load(save_filename))

    def save_model(self, save_filename):
        torch.save(self.state_dict(), save_filename)
        
    def mlabel_evaluate(src, tgt, future_size=1):
        pass
        
    def mlabel_predict(src, tgt, future_size=1):
        with torch.no_grad():
            seq = torch.cat((src, tgt), dim=1)
            dummy_point = torch.zeros((seq.size(0), 1, seq.size(2))).to(self.device)
            
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
        return seq[:, -future_size:]

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
    
    def show_classification(self, src, tgt, plot_folder='', plot_name='transformer_classifier_results.png'):
        plot_path = os.path.join(os.getcwd(), plot_folder, plot_name)
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
        plt.savefig(plot_path)
        plt.show()
        
    def show_example(self, src, tgt, ftr, plot_folder='', plot_name='transformer_results.png'):
        plot_path = os.path.join(os.getcwd(), plot_folder, plot_name)
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
        plt.savefig(plot_path)
        plt.show()
        
    def plot_loss(self, plot_folder='', plot_name='transformer_loss.png'):
        plot_path = os.path.join(os.getcwd(), plot_folder, plot_name)
        epoch_array = np.arange(self.epochs)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(epoch_array, self.mean_train_loss, label='train')
        ax.plot(epoch_array, self.mean_test_loss, label='test')
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')
        ax.legend(loc='upper right')
        plt.savefig(plot_path)
        plt.show()