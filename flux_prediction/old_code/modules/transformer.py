###### import modules ######

import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
#from torchmetrics import F1Score
from transformers import BertForSequenceClassification
import pandas as pd
import seaborn as sns
from typing import Tuple


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


class F1Score:
    """
    Class for f1 calculation in Pytorch.
    """

    def __init__(self, average: str = 'weighted', requires_grad: bool = True, device=torch.device("cpu")):
        """
        Init.

        Args:
            average: averaging method
        """
        self.average = average
        self.requires_grad = requires_grad
        self.device=device
        if average not in [None, 'micro', 'macro', 'weighted']:
            raise ValueError('Wrong value of average parameter')

    #@staticmethod
    def calc_f1_micro(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 micro.

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels

        Returns:
            f1 score
        """
        true_positive = torch.eq(labels, predictions).sum().float().to(self.device)
        f1_score = torch.div(true_positive, len(labels))
        if self.requires_grad:
            f1_score.requires_grad = True
        return f1_score

    #@staticmethod
    def calc_f1_count_for_label(self, predictions: torch.Tensor,
                                labels: torch.Tensor, label_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate f1 and true count for the label

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels
            label_id: id of current label

        Returns:
            f1 score and true count for label
        """
        # label count
        true_count = torch.eq(labels, label_id).sum().to(self.device)

        # true positives: labels equal to prediction and to label_id
        true_positive = torch.logical_and(torch.eq(labels, predictions),
                                          torch.eq(labels, label_id)).sum().float().to(self.device)
        # precision for label
        precision = torch.div(true_positive, torch.eq(predictions, label_id).sum().float()).to(self.device)
        # replace nan values with 0
        precision = torch.where(torch.isnan(precision),
                                torch.zeros_like(precision).type_as(true_positive),
                                precision).to(self.device)

        # recall for label
        recall = torch.div(true_positive, true_count)
        # f1
        f1 = 2 * precision * recall / (precision + recall)
        # replace nan values with 0
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1).type_as(true_positive), f1)
        if self.requires_grad:
            f1_score.requires_grad = True
            #true_count.requires_grad = True
        return f1, true_count

    def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 score based on averaging method defined in init.

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels

        Returns:
            f1 score
        """

        # simpler calculation for micro
        if self.average == 'micro':
            return self.calc_f1_micro(predictions, labels)

        f1_score = 0
        for label_id in range(len(labels.unique())):
            f1, true_count = self.calc_f1_count_for_label(predictions, labels, label_id)

            if self.average == 'weighted':
                f1_score += f1 * true_count
            elif self.average == 'macro':
                f1_score += f1

        if self.average == 'weighted':
            f1_score = torch.div(f1_score, len(labels)).to(self.device)
        elif self.average == 'macro':
            f1_score = torch.div(f1_score, len(labels.unique())).to(self.device)
            
        if self.requires_grad:
            f1_score.requires_grad = True

        return (1 - f1_score)

"""
def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)
"""

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

class FlareClassificationTransformer(nn.Module):
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

    def __init__(self, train_dset, test_dset, #multilabel=False,
                 model_d=32, nheads=1, encoding='pos', time_encoding_dim=8, 
                 f1_loss=False, optimizer='adam', bert=False,
                 enc_layers=1, dec_layers=1, prediction_distance=1, epochs=100,
                 batch_size=64, learning_rate=1e-4, gamma=0.99, device=torch.device("cpu")):
        
        super(FlareClassificationTransformer, self).__init__()
        #self.multilabel = multilabel
        self.f1_loss = f1_loss
        self.epochs = epochs
        self.device = device
        self.prediction_distance = prediction_distance
        self.bert = bert

        #Load datasets
        self.train_data = DataLoader(train_dset, batch_size=batch_size)
        self.test_data = DataLoader(test_dset, batch_size=batch_size)

        # Store some dimensions of the data
        self.sequence_length = train_dset.get_sequence_length()
        self.target_size = train_dset.get_target_size()
        self.data_in_dim = train_dset.get_n_features()
        """
        if not self.multilabel:
            self.data_out_dim = self.data_in_dim
            self.future_size = train_dset.get_future_size()
            self.source_size = self.sequence_length - self.target_size - self.future_size #?????
        else:
        """
        self.source_size = 10
        self.data_out_dim = self.data_in_dim
        self.n_labels = train_dset.get_n_labels()

        # Feature dimension the transformer will use
        self.model_dim = model_d
        
        #set time encoding dim
        self.time_encoding_dim = time_encoding_dim

        # Positional encoding type
        self.encoding = encoding
        self.positional_encoding, self.delay_buffer = self.set_encoding(encoding, data_dim=self.data_in_dim, model_dim=self.model_dim,
                                                                        time_encoding_dim=self.time_encoding_dim, device=self.device)
        
        # Transformer model (pytorch's implementation of the transformer)
        """
        if not self.multilabel:
            # Mask for the target
            self.tgt_mask = self._generate_square_subsequent_mask(self.target_size).to(self.device)
            self.transformer = nn.Transformer(d_model=self.model_dim, nhead=nheads, num_encoder_layers=enc_layers,
                                              num_decoder_layers=dec_layers, dropout=0.2, batch_first=True)
            self.decoder = nn.Linear(self.model_dim, self.data_out_dim)
            self.loss_fn = nn.MSELoss(reduction='mean')
            self.loss_str = 'mse'
        else:
        """
        #min and max value of inputs, needed for multilabel masking
        self.minvalue = train_dset.get_minvalue()
        self.maxvalue = train_dset.get_maxvalue()

        #no target mask needed (target is not used on forward)
        self.tgt_mask = None
        flat_dim = self.model_dim*self.source_size
        dense1_out_dim = self.source_size * self.data_out_dim

        #transformer
        if self.bert:
            bert_name = 'bert-mini_bert_params_' + str(self.n_labels)
            self.transformer = BertForSequenceClassification.from_pretrained(os.path.join(os.getcwd(), 'model_params', bert_name))

            #for param in self.transformer.bert.parameters():
                #param.requires_grad = False
            #for w, b in zip(self.transformer.bert.classifier.weight, self.transformer.bert.classifier.bias):
                #w.requires_grad = False
                #b.requires_grad = False
            #self.resize_layer = nn.Linear([16, 10, 64], [16, 10, 768])
        else:
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=nheads, dropout=0.2,
                                                            batch_first=True)
            self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=enc_layers)

            #classifier
            self.dropout = nn.Dropout(0.1)
            self.flatten = nn.Flatten(start_dim=1)
            self.dense1 = nn.Linear(flat_dim, flat_dim)
            self.activ1 = nn.Tanh()
            self.dense2 = nn.Linear(flat_dim, flat_dim)
            self.activ2 = nn.LeakyReLU()
            self.norm = nn.LayerNorm(flat_dim)
            self.classifier = nn.Linear(flat_dim, self.n_labels)

            self.softmax = nn.Softmax(dim=1) #needed only for evaluation

        if self.f1_loss:
            self.loss_fn = F1Score(average='macro', device=device)
            #self.loss_fn = F1Score(num_classes=self.n_labels, average='macro')
            self.loss_str = '1 - f1 score'
        else:
            self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
            self.loss_str = 'cross entropy'
        
        if not self.bert:
            # weight initialization (I just copied this, don't know if it helps much)
            self.init_weights()

        # Loss, optimizer and learning rate scheduler
        self.optim_str = optimizer
        if self.optim_str == 'adam':
            self.optimizer = optim.Adam([{'params': self.parameters()}], lr=learning_rate, weight_decay=1e-5)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=gamma)
        elif self.optim_str == 'adamw':
            self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, eps=1e-6)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=gamma)
        elif self.optim_str == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma)
            
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=1e-4, max_lr=0.1, step_size_up=100)
        
        #loss per epoch
        self.mean_train_loss = []
        self.mean_test_loss = []
        self.train_f1 = []
        self.test_f1 = []
        
    def set_encoding(self, encoding, data_dim=None, model_dim=None, time_encoding_dim=None, device=None):
        if encoding == 'pos':
            positional_encoding = nn.Sequential(nn.Linear(data_dim, model_dim), 
                                                PositionalEncoding(model_dim))
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
        if self.multilabel:
            if self.bert:
                print(dir(self.transformer.bert))
                self.transformer.bert.data.zero_()
                self.transformer.bert.data.uniform_(-initrange, initrange)
            else:
                self.dense1.bias.data.zero_()
                self.dense1.weight.data.uniform_(-initrange, initrange)
                self.dense2.bias.data.zero_()
                self.dense2.weight.data.uniform_(-initrange, initrange)
        else:
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt=None):
        """
        if not self.multilabel:
            # Concatenate the source and target
            seq = torch.cat((src, tgt), dim=1)
        else:
        """
        seq = src
        #print(seq.shape)

        # Add positional encoding to each point in the sequence (this also changes the feature dimension to d_model)
        seq = self.positional_encoding(seq)

        # Use the (source_size) first steps as the source
        """
        if not self.multilabel:
            src = seq[:, :self.source_size - self.delay_buffer]

            # (!) Use the points (target_size - 1) until the second to last point as the decoder input
            decoder_input = seq[:, -self.target_size - self.prediction_distance :-self.prediction_distance]

            # Transformer model magic (out has the same shape as decoder_input)
            # The tgt_mask is the mask for decoder_input
            output = self.transformer(src, decoder_input, tgt_mask=self.tgt_mask)
            output = self.decoder(output)
        else:
        """
        if self.bert:
            #output = self.resize_layer(
            output = self.transformer(inputs_embeds=seq)
            output = output.logits[:, None, :]
        else:
            #src_mask = self._generate_dynamic_random_mask(self.source_size)
            src_mask = None
            output = self.transformer(seq, src_mask)

        # Part of alternative model
        # out = self.transformer_encoder(src, self.src_mask)
        # out = out[:, -self.target_size:]

        # Change the feature dimension back to [flux, area]
        #if self.multilabel and not self.bert:
        if not self.bert:
            output = self.dropout(output)
            output = self.flatten(output)
            output = self.dense1(output)
            output = self.activ1(output)
            output = self.dense2(output)
            output = self.activ2(output)
            output = self.norm(output)
            output = self.classifier(output)
            
        if self.bert:
            output = output[:, 0]
        if self.f1_loss and not self.bert:
            output = self.predict_labels(output)
            
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)
        return mask
    
    def _generate_dynamic_random_mask(self, size, masked=0.15, masked_ratios=[0.8, 0.1, 0.1]):
        """
        masks the sequence values, masked values are 15%
        of these values 80% are set to -inf, 10% to a random value and 10% to the actual value
        """
        n_elements = size**2
        n_idx = int(masked*n_elements)
        idx_l = [(i,j) for i, j in zip(torch.randperm(size), torch.randperm(size))]
        idx_l = idx_l[:n_idx]

        mask = torch.full((size, size), 1.).to(self.device)
        for idx in idx_l:
            if torch.rand(1) < 0.8:
                mask[idx] = float('-inf')
            else:
                if torch.rand(1) < 0.5:
                    mask[idx] = torch.FloatTensor(1).uniform_(self.minvalue, self.maxvalue)
                else:
                    pass
        
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
            
            train_pred_lbl_list = []
            train_tgt_lbl_list = []

            ###### Training ######
            
            #if self.multilabel:
            print(f'+++ Epoch {epoch+1} +++')
            for (src, tgt) in tqdm(self.train_data, desc='Training ', total=n_train_batches, ascii=' >='):
                self.optimizer.zero_grad()
                output = self(src)
                tgt = self.labels_from_target(tgt)

                loss = self.loss_fn(output, tgt)

                tmp_loss = loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optimizer.step()
                train_loss.append(tmp_loss)
                
                #labels for f1 score
                pred_labels = self.predict_labels(output)
                train_tgt_lbl_list.extend(tgt.tolist())
                train_pred_lbl_list.extend(pred_labels.tolist())
                
            """
                tmp_loss = loss.item()
                #if self.f1_loss:
                #    tmp_loss = loss.item() + 1
                train_loss.append(tmp_loss)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optimizer.step()
                    
            else:
                for (src, tgt, _) in tqdm(self.train_data, desc=f'Epoch {epoch}', total=n_train_batches, ascii=' >='):
                    self.optimizer.zero_grad()
                    output = self(src, tgt=tgt)
                    loss = self.loss_fn(output, tgt) #change to 
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                    self.optimizer.step()
                    train_loss.append(loss.item())
            """
            
            current_mean_train_loss = np.mean(train_loss)
            self.mean_train_loss.append(current_mean_train_loss)
            current_train_f1 = f1_score(train_tgt_lbl_list, train_pred_lbl_list, average='macro')
            self.train_f1.append(current_train_f1)
            
            ###### Evaluating ######

            self.eval()
            test_loss = []
            
            test_pred_lbl_list = []
            test_tgt_lbl_list = []

            # Validation
            with torch.no_grad():
                #if self.multilabel:
                for (src, tgt) in tqdm(self.test_data, desc='Testing  ', total=n_test_batches, ascii=' >='):
                    output = self(src)
                    tgt = self.labels_from_target(tgt)

                    loss = self.loss_fn(output, tgt)
                    
                    tmp_loss = loss.item()
                    test_loss.append(tmp_loss)
                    
                    #labels for f1 score
                    pred_labels = self.predict_labels(output)
                    test_tgt_lbl_list.extend(tgt.tolist())
                    test_pred_lbl_list.extend(pred_labels.tolist())
                """
                else:
                    for (src, tgt, _) in tqdm(self.test_data, desc='Testing  ', total=n_test_batches, ascii=' >='):
                        output = self(src, tgt)
                        loss = self.loss_fn(output, tgt)
                        test_loss.append(loss.item())
                """

            current_mean_test_loss = np.mean(test_loss)
            self.mean_test_loss.append(current_mean_test_loss)
            current_test_f1 = f1_score(test_tgt_lbl_list, test_pred_lbl_list, average='macro')
            self.test_f1.append(current_test_f1)
            
            print(f'({self.loss_str})| Train loss:', current_mean_train_loss, '| Test loss:', current_mean_test_loss)
            print('Train f1:', current_train_f1, '| Test f1:', current_test_f1, '| Learning rate: ', self.scheduler.get_last_lr()[0])
            print('')

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
    
    def predict_labels(self, prob):
        label_idx = torch.argmax(prob, dim=1)
        return label_idx
    """
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
    
    def predict_labels(self, pred_tgt):
        if not self.bert:
            pred_tgt = self.softmax(pred_tgt)
            dim=1
        else:
            dim=2
            
        label_idx = torch.argmax(pred_tgt, dim=dim)
        
        if self.bert:
            label_idx = label_idx[:, 0]
        
        return label_idx
    """
    
    def labels_from_target(self, tgt):
        tgt = torch.argmax(tgt, dim=2)
        tgt = torch.flatten(tgt)
        return tgt
    
    def class_report(self, tgt_labels, pred_labels):
        rep_str = classification_report(tgt_labels, pred_labels)#, output_dict=True)
        return rep_str 
    
    def calc_f1(self, tgt_labels, pred_tgt):
        pred_labels = self.predict_labels(pred_tgt)
        f1 = f1_score(tgt_labels.detach().cpu(), pred_labels.detach().cpu())
        return f1
    
    def plot_confusion_matrix(self, eval_data, plot_folder='', plot_name='transformer_confusion_matrix.png'):
        plot_path = os.path.join(os.getcwd(), plot_folder, plot_name)
        label_idx_list = []
        tgt_list = []
        
        self.eval()
        with torch.no_grad():
            for (src, tgt) in tqdm(eval_data, desc='Evaluating ', total=len(eval_data), ascii=' >='):
                pred_tgt = self(src)
                tgt = self.labels_from_target(tgt)
                #if not self.f1_loss:
                label_idx = self.predict_labels(pred_tgt)

                tgt_list.extend(tgt.tolist())
                label_idx_list.extend(label_idx.tolist())
            
        cmt = confusion_matrix(tgt_list, label_idx_list, normalize='true')
        rep_str = self.class_report(tgt_list, label_idx_list)
        print('+++ Model evaluation +++')
        print(rep_str)
        
        if self.n_labels == 5:
            df_cm = pd.DataFrame(cmt, index = ['No flare', 'C', 'B', 'M', 'X'],
                                 columns = ['No flare', 'C', 'B', 'M', 'X'])
        elif self.n_labels == 3:
            df_cm = pd.DataFrame(cmt, index = ['No flare', 'C+B', 'M+X'],
                                 columns = ['No flare', 'C+B', 'M+X'])
        elif self.n_labels == 2:
            df_cm = pd.DataFrame(cmt, index = ['No flare', 'Flare'],
                                 columns = ['No flare', 'Flare'])
        plt.figure(figsize = (10,7))
        sns.heatmap(df_cm, cmap="Blues", annot=True)
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
        """
        if self.bert:
            epoch_array = np.arange(10)
            self.mean_train_loss.extend([i for i in range(10)])
            self.mean_test_loss.extend([i for i in range(10)])
        else:
        """
        epoch_array = np.arange(self.epochs)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(epoch_array, self.mean_train_loss, label='train loss')
        ax.plot(epoch_array, self.mean_test_loss, label='test loss')
        ax.plot(epoch_array, self.train_f1, label='train f1')
        ax.plot(epoch_array, self.test_f1, label='test f1')
       
        #if self.multilabel:
        ax.set_title('{} labels, optimizer: {}'.format(self.n_labels, self.optim_str))
            
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss: ' + self.loss_str)
        ax.legend(loc='upper right')
        plt.savefig(plot_path)
        plt.show()