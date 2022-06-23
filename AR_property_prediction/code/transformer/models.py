import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertModel
from .datasets import SinusoidDataset, PolynomialDataset, FluxDataset
from .encoding import PositionalEncoding, SineActivation, TimeDelay


class ExtractH(nn.Module):
    """
    Used for extracting the hidden state from a pytorch LSTM output
    """
    def forward(self, x):
        tensor, _ = x
        return tensor


class LSTMModel(nn.Module):
    """
    Pytorch version of LSTM model for flux prediction.
    """

    def __init__(self, feature_dim, target_size):
        super().__init__()
        self.encoder = nn.Sequential(nn.LSTM(feature_dim, 256, batch_first=True),
                                     ExtractH(),
                                     nn.LSTM(256, 128, batch_first=True),
                                     ExtractH(),
                                     nn.LSTM(128, 64, batch_first=True),
                                     ExtractH(),
                                     nn.LSTM(64, 64, batch_first=True),
                                     ExtractH(),
                                     nn.LSTM(64, 64, batch_first=True),
                                     ExtractH())
        self.decoder = nn.Sequential(nn.LSTM(64, 256, batch_first=True),
                                     ExtractH(),
                                     nn.LSTM(256, 128, batch_first=True),
                                     ExtractH(),
                                     nn.LSTM(128, 64, batch_first=True),
                                     ExtractH(),
                                     nn.Linear(64, feature_dim))

        self.target_size = target_size

    def forward(self, x):
        out = self.encoder(x)
        out = out[:, -1].unsqueeze(1).repeat(1, self.target_size, 1)
        out = self.decoder(out)
        return out


def weighted_mse_loss(output_size, device):
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

    def weighted_mse(outputs, targets):
        return torch.mean(weights * (outputs[..., :1] - targets[..., :1]) ** 2)

    return weighted_mse


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

    def __init__(self, data_dir, set_id=None, model_id=None, source_size=100, target_size=25, future_size=25,
                 model_d=32, nheads=1, encoding='pos', time_encoding_dim=8, enc_layers=1, dec_layers=1, epochs=100,
                 batch_size=256, learning_rate=1e-4, gamma=0.99, use_baseline=False, device=torch.device("cpu")):
        super(FluxTransformer, self).__init__()
        self.epochs = epochs
        self.device = device
        self.baseline = use_baseline
        self.model_id = model_id

        # Create datasets
        if set_id == 'sin':
            self.train_data = DataLoader(SinusoidDataset(n_sequences=800, source_size=source_size,
                                                         target_size=target_size, future_size=future_size,
                                                         add_noise=True, device=self.device), batch_size=batch_size)
            self.test_data = DataLoader(SinusoidDataset(n_sequences=200, source_size=source_size,
                                                        target_size=target_size, future_size=future_size,
                                                        add_noise=True, device=self.device), batch_size=batch_size)
        elif set_id == 'pol':
            self.train_data = DataLoader(PolynomialDataset(n_sequences=800, source_size=source_size,
                                                           target_size=target_size, future_size=future_size,
                                                           add_noise=True, device=self.device), batch_size=batch_size)
            self.test_data = DataLoader(PolynomialDataset(n_sequences=200, source_size=source_size,
                                                          target_size=target_size, future_size=future_size,
                                                          add_noise=True, device=self.device), batch_size=batch_size)
        else:
            self.train_data = DataLoader(FluxDataset(data_dir, set_idx=set_id, source_size=source_size,
                                                     target_size=target_size,future_size=future_size, test=False,
                                                     device=self.device), batch_size=batch_size)
            self.test_data = DataLoader(FluxDataset(data_dir, set_idx=set_id, source_size=source_size,
                                                    target_size=target_size, future_size=future_size, test=True,
                                                    device=self.device), batch_size=batch_size)

        # Store some dimensions of the data
        self.target_size = target_size
        self.future_size = future_size
        self.source_size = source_size
        self.data_dim = self.train_data.dataset.n_features
        print('source size:', self.source_size)
        print('target size:', self.target_size)
        print('future size:', self.future_size)
        print('number of features:', self.data_dim)

        # Feature dimension the model will use
        self.model_dim = model_d

        # Positional encoding type
        self.encoding = encoding
        if encoding == 'pos':
            self.positional_encoding = nn.Sequential(nn.Linear(self.data_dim, self.model_dim),
                                                     PositionalEncoding(self.model_dim))
            self.delay_buffer = 0
        elif encoding == 't2v':
            self.positional_encoding = nn.Sequential(SineActivation(time_encoding_dim, self.device),
                                                     nn.Linear(time_encoding_dim + self.data_dim, self.model_dim))
            self.delay_buffer = 0
        elif encoding == 'td':
            self.positional_encoding = nn.Sequential(TimeDelay(dim=time_encoding_dim, delay=1),
                                                     nn.Linear(time_encoding_dim * self.data_dim, self.model_dim))
            self.delay_buffer = time_encoding_dim

        # Model type
        if model_id == 'transformer_encoder':
            # Use only the Transformer encoder
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=nheads, dropout=0.2,
                                                            batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=enc_layers)
            self.decoder = nn.Linear(self.model_dim * (self.source_size - self.delay_buffer), 1 * self.target_size)
            self.init_weights()
        elif model_id == 'full_transformer':
            # Uses both the encoder and (masked) decoder from the original Transformer
            self.tgt_mask = self._generate_square_subsequent_mask(self.target_size).to(self.device)
            self.transformer = nn.Transformer(d_model=self.model_dim, nhead=nheads, num_encoder_layers=enc_layers,
                                              num_decoder_layers=dec_layers, dropout=0.2, batch_first=True)
            self.decoder = nn.Linear(self.model_dim, self.data_dim)
        elif model_id == 'pretrained_bert':
            # Model with a pre trained bert module
            encoder_save_file = os.path.join(os.getcwd(), 'bert_encoder_params')
            decoder_save_file = os.path.join(os.getcwd(), 'bert_decoder_params')
            # model_name = 'bert-base-uncased'
            model_name = 'prajjwal1/bert-mini'
            if not os.path.exists(encoder_save_file):
                bert_encoder = BertModel.from_pretrained(model_name,
                                                         problem_type='regression')
                bert_encoder.save_pretrained(encoder_save_file)
                del bert_encoder
            if not os.path.exists(decoder_save_file):
                bert_decoder = BertModel.from_pretrained(model_name,
                                                         is_decoder=True,
                                                         add_cross_attention=True,
                                                         problem_type='regression')
                bert_decoder.save_pretrained(decoder_save_file)
                del bert_decoder
            self.bert_encoder = BertModel.from_pretrained(encoder_save_file)
            for param in self.bert_encoder.parameters():
                param.requires_grad = False
            self.bert_decoder = BertModel.from_pretrained(decoder_save_file)
            for param in self.bert_decoder.parameters():
                param.requires_grad = False
            mask = torch.triu(torch.full((self.target_size, self.target_size), 1), diagonal=1)
            self.tgt_mask = torch.stack(batch_size * [mask], dim=0)  # [batch, tgt_size, tgt_size]
            self.decoder = nn.Linear(self.model_dim, self.data_dim)
        elif model_id == 'lstm':
            # For testing a pytorch lstm
            self.lstm = LSTMModel(self.data_dim, self.target_size)

        # Loss, optimizer and learning rate scheduler
        self.loss_fn = nn.MSELoss(reduction='mean')
        # self.loss_fn = weighted_mse_loss(self.future_size, device)

        self.optimizer = optim.NAdam([{'params': self.parameters()}], lr=learning_rate)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=gamma)

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt):
        baseline = src[:, -1:, :1]

        if self.model_id == 'full_transformer':
            seq = torch.cat((src, tgt), dim=1)
            seq = self.positional_encoding(seq)
            src = seq[:, :self.source_size - self.delay_buffer]
            decoder_input = seq[:, -self.target_size - 1: -1]

            out = self.transformer(src, decoder_input, tgt_mask=self.tgt_mask)
            out = self.decoder(out)
        elif self.model_id == 'transformer_encoder':
            src = self.positional_encoding(src)
            src = src[:, :self.source_size - self.delay_buffer]

            out = self.transformer_encoder(src)
            out = out.reshape(out.size(0), -1).unsqueeze(1)
            out = self.decoder(out)
            out = out.reshape(out.size(0), self.target_size, -1)
        elif self.model_id == 'pretrained_bert':
            if src.size(0) != self.tgt_mask.size(0):
                mask = torch.triu(torch.full((self.target_size, self.target_size), 1), diagonal=1)
                mask = torch.stack(src.size(0) * [mask], dim=0)  # [batch, tgt_size, tgt_size]
            else:
                mask = self.tgt_mask
            seq = torch.cat((src, tgt), dim=1)
            seq = self.positional_encoding(seq)
            src = seq[:, :self.source_size - self.delay_buffer]
            decoder_input = seq[:, -self.target_size - 1: -1]

            out = self.bert_encoder(inputs_embeds=src)
            out = self.bert_decoder(inputs_embeds=decoder_input,
                                    encoder_hidden_states=out.last_hidden_state,
                                    attention_mask=mask)
            out = self.decoder(out.last_hidden_state)
        elif self.model_id == 'lstm':
            out = self.lstm(src)
        else:
            out = torch.zeros_like(tgt)

        if self.baseline:
            out += baseline

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
            for (src, tgt, ftr) in tqdm(self.train_data, desc=f'Epoch {epoch}', total=n_train_batches):
                self.optimizer.zero_grad()
                output = self(src, tgt)
                if self.future_size == 0:
                    tgt = tgt[:, :, :1]
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
            for (src, tgt, ftr) in tqdm(self.test_data, desc=f'Epoch {epoch}', total=n_test_batches):
                output = self(src, tgt)
                if self.future_size == 0:
                    tgt = tgt[:, :, :1]
                loss = self.loss_fn(output, tgt)
                test_loss.append(loss.item())

            mean_test_loss = np.mean(test_loss)
            print('Test loss:', mean_test_loss)

            # Check if the validation loss is smaller than it was for any previous epoch
            if mean_test_loss < min_loss:
                self.save_model(save_filename)
                min_loss = mean_test_loss

            # Adjust the learning rate
            self.scheduler.step()

    def load_model(self, save_filename, device=torch.device('cpu')):
        self.load_state_dict(torch.load(save_filename, map_location=device))

    def save_model(self, save_filename):
        torch.save(self.state_dict(), save_filename)

    def predict(self, src, tgt):
        # Predict future steps by repeatedly adding a dummy point to the sequence and predicting its value
        seq = torch.cat((src, tgt), dim=1)
        dummy_point = torch.zeros((seq.size(0), 1, seq.size(2))).to(self.device)
        for _ in range(self.future_size):
            seq = torch.cat((seq, dummy_point), dim=1)
            src = seq[:, -self.source_size - self.target_size:-self.target_size]
            tgt = seq[:, -self.target_size:]
            out = self(src, tgt)
            seq[:, -1:] = out[:, -1:]
        return seq[:, -self.future_size:]

    def loss(self, src, tgt, ftr, baseline=False):
        self.eval()
        if self.future_size == 0:
            if not baseline:
                pred_tgt = self(src, tgt)
                loss = self.loss_fn(tgt[:, :, :1], pred_tgt[:, :, :1]).item()
            else:
                loss = self.loss_fn(tgt[:, :, :1], torch.stack(tgt.size(1) * [src[:, -1, :1]], dim=1)).item()
        else:
            if not baseline:
                pred_ftr = self.predict(src, tgt)
                loss = self.loss_fn(ftr[:, :, :1], pred_ftr[:, :, :1]).item()
            else:
                loss = self.loss_fn(ftr[:, :, :1], torch.stack(ftr.size(1) * [tgt[:, -1, :1]], dim=1)).item()
        return loss

    def show_example(self, src, tgt, ftr):
        self.eval()

        fig, ax = plt.subplots(3, 3, figsize=(16, 12))

        # Prediction of the target
        pred_tgt = self(src, tgt)

        # Timesteps
        src_t = np.arange(self.source_size + 1)
        ftr_t = np.arange(self.source_size + self.target_size, self.source_size + self.target_size + self.future_size)

        if self.future_size != 0:
            # Target timesteps
            tgt_t = np.arange(self.source_size, self.source_size + self.target_size + 1)

            # Prediction further into the future
            with torch.no_grad():
                pred_ftr = self.predict(src, tgt)

            # Concatenate last step of target and future for smooth plot
            src = torch.cat((src, tgt[:, :1, :]), dim=1)
            tgt = torch.cat((tgt, ftr[:, :1, :]), dim=1)
            pred_tgt = torch.cat((pred_tgt, pred_ftr[:, :1, :]), dim=1)
        else:
            # Target timesteps
            tgt_t = np.arange(self.source_size, self.source_size + self.target_size)

            # Concatenate last step of target for smooth plot
            src = torch.cat((src, tgt[:, :1, :]), dim=1)

        for i in range(3):
            for j in range(3):
                ax[i, j].plot(src_t, src[3 * i + j, :, 0].cpu().detach().numpy(), c='black', label='Source')
                ax[i, j].plot(tgt_t, tgt[3 * i + j, :, 0].cpu().detach().numpy(), '--', c='black', label='Target')
                ax[i, j].plot(tgt_t, pred_tgt[3 * i + j, :, 0].cpu().detach().numpy(), '--', c='red',
                              label='Prediction')
                if self.future_size != 0:
                    ax[i, j].plot(ftr_t, ftr[3 * i + j, :, 0].cpu().detach().numpy(), label='future')
                    ax[i, j].plot(ftr_t, pred_ftr[3 * i + j, :, 0].cpu().detach().numpy(), label='future_prediction')
                ax[i, j].set_ylabel('Magnetic Flux (Mx)', fontsize=12)
                # ax[i, j].set_ylim(-2.5, 2.5)
                ax[i, j].legend(frameon=False, fontsize=12)
                ax[i, j].set_xlabel(r'Time ($\times 60$min)', fontsize=12)

                ax[i, j].minorticks_on()
                ax[i, j].tick_params(which='both', bottom=True, top=True, left=True, right=True)
                ax[i, j].tick_params(which='major', length=8, direction='in', labelsize=10)
                ax[i, j].tick_params(which='minor', length=4, direction='in', labelsize=10)
        plt.tight_layout()
        plt.savefig('results.png')
        plt.show()
