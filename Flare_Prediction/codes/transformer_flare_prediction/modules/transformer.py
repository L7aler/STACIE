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

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

rfont = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rfont)
plt.rcParams["legend.labelspacing"] = 0.001
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


###### define loss functions ######

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

        return 1 - f1_score


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

    def __init__(self, train_dset, test_dset,
                 model_dim=32, nheads=1, encoding='pos', time_encoding_dim=8, 
                 f1_loss=False, optimizer='adam', bert=False,
                 enc_layers=1, epochs=100,
                 batch_size=64, learning_rate=1e-4, gamma=0.99, 
                 device=torch.device("cpu")):
        
        super(FlareClassificationTransformer, self).__init__()
        self.f1_loss = f1_loss
        self.device = device
        self.bert = bert
        self.epochs = epochs
        self.prediction_distance = 1
        
        # transformer parameters
        self.nheads = nheads
        self.enc_layers = enc_layers

        #Load datasets
        self.train_data = DataLoader(train_dset, batch_size=batch_size)
        self.test_data = DataLoader(test_dset, batch_size=batch_size)

        # Store some dimensions of the data
        self.sequence_length = train_dset.get_sequence_length()
        self.target_size = train_dset.get_target_size()
        self.data_in_dim = train_dset.get_n_features()
        self.source_size = 10
        self.data_out_dim = self.data_in_dim
        self.n_labels = train_dset.get_n_labels()

        # Feature dimension the transformer will use
        self.model_dim = model_dim
        
        #set time encoding dim
        self.time_encoding_dim = time_encoding_dim

        # Positional encoding type
        self.encoding = encoding
        self.positional_encoding, self.delay_buffer = self.set_encoding(encoding, data_dim=self.data_in_dim, model_dim=self.model_dim,
                                                                        time_encoding_dim=self.time_encoding_dim, device=self.device)
        
        #min and max value of inputs, needed for multilabel masking
        self.minvalue = train_dset.get_minvalue()
        self.maxvalue = train_dset.get_maxvalue()

        # Transformer model (pytorch's implementation of the transformer)
        if self.bert:
            bert_name = 'bert-mini_bert_params_' + str(self.n_labels)
            self.transformer = BertForSequenceClassification.from_pretrained(os.path.join(os.getcwd(), 'model_params', bert_name))
        else:   
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=self.nheads, 
                                                            dropout=0.2, batch_first=True)
            self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=self.enc_layers)

            #classifier
            flat_dim = self.model_dim*self.source_size
            
            self.dropout1 = nn.Dropout(0.2)
            self.flatten = nn.Flatten(start_dim=1)
            self.dense1 = nn.Linear(flat_dim, flat_dim)
            self.activ1 = nn.Tanh()
            self.dropout2 = nn.Dropout(0.2)
            self.dense2 = nn.Linear(flat_dim, flat_dim)
            self.activ2 = nn.LeakyReLU()
            self.dropout3 = nn.Dropout(0.2)
            self.norm = nn.LayerNorm(flat_dim)
            self.classifier = nn.Linear(flat_dim, self.n_labels)

        self.softmax = nn.Softmax(dim=1) #needed only for evaluation

        if self.f1_loss:
            self.loss_fn = F1Score(average='macro', device=device)
            self.loss_str = '1 - f1 score'
        else:
            crossentr_weights = torch.Tensor(train_dset.get_class_weights()).to(self.device)
            self.loss_fn = nn.CrossEntropyLoss(weight=crossentr_weights, reduction='mean')
            self.loss_str = 'cross entropy'
        
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
        
        #saved as logs
        self.norm_cmt = None
        self.not_norm_cmt = None
        self.crep = None
        self.eval_tgt = None
        self.eval_out_labels = None
        self.eval_out_prob = None
        
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
        for name, param in self.named_parameters():
            try:
                param.bias.data.zero_()
                param.weight.data.uniform_(-initrange, initrange)
            except:
                print(name)

    def forward(self, src):
        # Add positional encoding to each point in the sequence (this also changes the feature dimension to d_model)
        seq = self.positional_encoding(src)

        if self.bert:
            output = self.transformer(inputs_embeds=seq, attention_mask=None)
            output = output.logits[:, None, :]
            output = output[:, 0]
        else:
            #src_mask = self._generate_dynamic_random_mask(self.source_size)
            src_mask = None
            output = self.transformer(seq, src_mask)
            output = self.dropout1(output)
            output = self.flatten(output)
            output = self.dense1(output)
            output = self.activ1(output)
            output = self.dropout2(output)
            output = self.dense2(output)
            output = self.activ2(output)
            output = self.dropout3(output)
            output = self.norm(output)
            output = self.classifier(output)
       
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
                    
    def train_model(self, save_filename=None, load_cp=False, save=False):
        if load_cp:
            if os.path.exists(save_filename):
                self.load_model(save_filename)
            else:
                print(f'Model not loaded. No model exists at {save_filename}')

        min_loss = np.inf

        n_train_batches = len(self.train_data)
        n_test_batches = len(self.test_data)
        
        for epoch in range(self.epochs):
            print(f'+++ Epoch {epoch+1} +++')
            
            ###### Training ######
            
            self.train()
            train_loss = []
            train_pred_lbl_list = []
            train_tgt_lbl_list = []
            
            for (src, tgt) in tqdm(self.train_data, desc='Training ', total=n_train_batches, ascii=' >='):
                self.optimizer.zero_grad()
                output = self(src)
                tgt = self.labels_from_target(tgt)
                if self.f1_loss:
                    output = self.predict_labels(output)

                loss = self.loss_fn(output, tgt)

                tmp_loss = loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optimizer.step()
                train_loss.append(tmp_loss)
                
                #labels for f1 score
                if self.f1_loss:
                    pred_labels = output
                else:
                    pred_labels = self.predict_labels(output)
                train_tgt_lbl_list.extend(tgt.tolist())
                train_pred_lbl_list.extend(pred_labels.tolist())
            
            current_mean_train_loss = np.mean(train_loss)
            self.mean_train_loss.append(current_mean_train_loss)
            current_train_f1 = f1_score(train_tgt_lbl_list, train_pred_lbl_list, average='macro')
            self.train_f1.append(current_train_f1)
            
            ###### Evaluating ######

            self.eval()
            test_loss = []
            test_pred_lbl_list = []
            test_tgt_lbl_list = []

            with torch.no_grad():
                for (src, tgt) in tqdm(self.test_data, desc='Testing  ', total=n_test_batches, ascii=' >='):
                    output = self(src)
                    tgt = self.labels_from_target(tgt)
                    if self.f1_loss:
                        output = self.predict_labels(output)

                    loss = self.loss_fn(output, tgt)
                    
                    tmp_loss = loss.item()
                    test_loss.append(tmp_loss)
                    
                    #labels for f1 score
                    if self.f1_loss:
                        pred_labels = output
                    else:
                        pred_labels = self.predict_labels(output)
                    test_tgt_lbl_list.extend(tgt.tolist())
                    test_pred_lbl_list.extend(pred_labels.tolist())

            current_mean_test_loss = np.mean(test_loss)
            self.mean_test_loss.append(current_mean_test_loss)
            current_test_f1 = f1_score(test_tgt_lbl_list, test_pred_lbl_list, average='macro')
            self.test_f1.append(current_test_f1)
            
            print(f'({self.loss_str})| Train loss:', current_mean_train_loss, '| Test loss:', current_mean_test_loss)
            print('Train f1:', current_train_f1, '| Test f1:', current_test_f1, '| Learning rate:', self.scheduler.get_last_lr()[0])
            print('')

            # Adjust the learning rate
            self.scheduler.step()
            
        if save:
            self.save_model(save_filename)

    def load_model(self, save_filename):
        self.load_state_dict(torch.load(save_filename))

    def save_model(self, save_filename):
        torch.save(self.state_dict(), save_filename)
    
    def predict_labels(self, prob):
        label_idx = torch.argmax(prob, dim=1)
        return label_idx
    
    def labels_from_target(self, tgt):
        tgt = torch.argmax(tgt, dim=2)
        tgt = torch.flatten(tgt)
        return tgt
    
    def test_model(self, test_data):
        label_idx_list = []
        tgt_list = []
        prob_list = []
        
        self.eval()
        with torch.no_grad():
            for (src, tgt) in tqdm(test_data, desc='Evaluating ', total=len(test_data), ascii=' >='):
                pred_tgt = self(src)
                norm_prob = self.softmax(pred_tgt)
                tgt = self.labels_from_target(tgt)
                label_idx = self.predict_labels(norm_prob)

                tgt_list.extend(tgt.tolist())
                label_idx_list.extend(label_idx.tolist())
                prob_list.extend(norm_prob.tolist())
                
        rep_str = classification_report(tgt_list, label_idx_list)
                
        self.crep = classification_report(tgt_list, label_idx_list, output_dict=True)
        self.eval_tgt = tgt_list
        self.eval_out_labels = label_idx_list
        self.eval_out_prob = prob_list
        
        print(rep_str)
                
    def plot_confusion_matrix(self, plot_folder='', plot_name='', show=True):
        plot_path = os.path.join(os.getcwd(), plot_folder, plot_name+'_conf_matrix.png')
        
        norm_cmat = confusion_matrix(self.eval_tgt, self.eval_out_labels, normalize='true')
        not_norm_cmat = confusion_matrix(self.eval_tgt, self.eval_out_labels)
        
        norm_cmat = norm_cmat*100

        if self.n_labels == 5:
            cmat_new = np.copy(norm_cmat)
            cmat_new[:, 1], cmat_new[:, 2] =  norm_cmat[:, 2], norm_cmat[:, 1]

            norm_cmat = np.copy(cmat_new)
            cmat_new[1, :], cmat_new[2, :] = norm_cmat[2, :], norm_cmat[1, :]
            
            nn_cmat_new = np.copy(not_norm_cmat)
            nn_cmat_new[:, 1], nn_cmat_new[:, 2] =  not_norm_cmat[:, 2], not_norm_cmat[:, 1]

            not_norm_cmat = np.copy(nn_cmat_new)
            nn_cmat_new[1, :], nn_cmat_new[2, :] = not_norm_cmat[2, :], not_norm_cmat[1, :]
        else:
            cmat_new = norm_cmat
            nn_cmat_new = not_norm_cmat
        
        self.norm_cmt = cmat_new.tolist()
        self.not_norm_cmt = nn_cmat_new.tolist()
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f'Confusion matrix - {self.n_labels} Classes', fontsize = 15)

        ax = sns.heatmap(cmat_new, cmap="Blues", annot=True,
                         fmt='.3g', #only for run02-5l-bert
                         cbar_kws={'label': 'Accuracy (%)'}, annot_kws={"size": 12})

        ax.set_xlabel('')
        ax.set_ylabel('')

        if self.n_labels == 5:
            ax.xaxis.set_ticklabels(['No flare', 'B', 'C', 'M', 'X'], size=12)
            ax.yaxis.set_ticklabels(['No flare', 'B', 'C', 'M', 'X'], size=12)
        elif self.n_labels == 3:
            ax.xaxis.set_ticklabels(['No flare', 'B+C', 'M+X'], size=12)
            ax.yaxis.set_ticklabels(['No flare', 'B+C', 'M+X'], size=12)
        elif self.n_labels == 2:
            ax.xaxis.set_ticklabels(['No Flare','Flare'], size=12)
            ax.yaxis.set_ticklabels(['No Flare','Flare'], size=12)

        ax.figure.axes[-1].yaxis.label.set_size(12)
        
        plt.savefig(plot_path)
        if show:
            plt.show()
        
    def plot_roc(self, plot_folder='', plot_name='', show=True):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111)
        plot_path2 = os.path.join(os.getcwd(), plot_folder, plot_name+'_roc.png')
        
        prob_array = np.array(self.eval_out_prob)
        
        if self.n_labels == 2:
            #only one roc curve
            current_y_score = prob_array[:, 1] #according to documentation y_score has to be probabilities of the positive class
            fpr[0], tpr[0], _ = roc_curve(self.eval_tgt, current_y_score)
            roc_auc[0] = auc(fpr[0], tpr[0])
            fpr["micro"], tpr["micro"], _ = roc_curve(np.array(self.eval_tgt).ravel(), current_y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            ax.plot(fpr[0], tpr[0], color = 'black', label="ROC curve (area = %0.2f)" % roc_auc[0])

        else:
            print('Multiclass ROC curve not available')
            """
            #multiple roc curves
            classes = [i for i in range(self.n_labels)]

            lin_tgt = label_binarize(tgt_list, classes=classes)
            print(lin_tgt)
            #prob_array = np.array(prob_list)
            #lin_out = label_binarize(prob_list, classes=classes)

            for i in range(self.n_labels):
                current_y_true = lin_tgt[:, i]
                current_y_score = prob_array[:, i]
                print(current_y_true)
                fpr[i], tpr[i], _ = roc_curve(current_y_true, current_y_score)
                roc_auc[i] = auc(fpr[i], tpr[i])

            fpr["micro"], tpr["micro"], _ = roc_curve(lin_tgt.ravel(), prob_array.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.n_labels)]))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(self.n_labels):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= self.n_labels

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            # Plot all ROC curves
            label_micro = "micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"])
            label_macro = "macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"])

            ax.plot(fpr["micro"], tpr["micro"], label=label_micro, linestyle=":", linewidth=2*lw)
            ax.plot(fpr["macro"], tpr["macro"], label=label_macro, linestyle=":", lw=2*lw)

            #colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            #for i, color in zip(range(n_classes), colors):
            for i in range(self.n_labels):
                label_class = "ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i])
                ax.plot(fpr[i], tpr[i], lw=lw, label=label_class)
            """

        if self.bert:
            ax.set_title('ROC curve - BERT-mini', fontsize = 15)
        else:
            ax.set_title('ROC curve - AL-BERTO', fontsize = 15)
        ax.plot([0, 1], [0, 1], color = 'black', ls = '--', label = 'Random Classifier')
        
        ax.set_ylabel('True Positive Rate', fontsize = 15)
        ax.set_xlabel('False Positive Rate', fontsize = 15)
        ax.minorticks_on()
        ax.tick_params(which = 'both', bottom = True, top = True, left = True, right = True)
        ax.tick_params(which = 'major', length = 8, direction = 'in', labelsize = 12)
        ax.tick_params(which = 'minor', length = 4, direction = 'in', labelsize = 12)
        ax.legend(frameon = False, fontsize = 15)
        
        if self.n_labels == 2:
            plt.savefig(plot_path2)
            if show:
                plt.show()
        
    def plot_loss(self, plot_folder='', plot_name='transformer_loss.png', show=True):
        plot_path = os.path.join(os.getcwd(), plot_folder, plot_name)
        epoch_array = np.arange(self.epochs)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(epoch_array, self.mean_train_loss, label='train loss')
        ax.plot(epoch_array, self.mean_test_loss, label='test loss')
        ax.plot(epoch_array, self.train_f1, label='train f1')
        ax.plot(epoch_array, self.test_f1, label='test f1')
       
        ax.set_title('{} labels, optimizer: {}'.format(self.n_labels, self.optim_str))
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss: ' + self.loss_str)
        ax.legend(loc='upper right')
        
        plt.savefig(plot_path)
        if show:
            plt.show()
        
    def save_logs(self, log_folder='', log_name=''):
        #len of these entries is 4
        save_dict_1 = self.crep
        
        accuracy_row = dict()
        accuracy_row['precision'] = 0
        accuracy_row['recall'] = 0
        accuracy_row['f1-score'] = save_dict_1['accuracy']
        accuracy_row['support'] = save_dict_1['macro avg']['support']
        
        save_dict_1['accuracy'] = accuracy_row
        str_1 = 'class_report_'
        log_path_1 = os.path.join(os.getcwd(), log_folder, str_1+log_name+'_log.csv')
        
        #len of these entries is epochs
        save_dict_2 = dict()
        save_dict_2['train_crossentropy_loss'] = self.mean_train_loss
        save_dict_2['test_crossentropy_loss'] = self.mean_test_loss
        save_dict_2['train_f1_score'] = self.train_f1
        save_dict_2['test_f1_score'] = self.test_f1
        str_2 = 'loss_score_epoch_'
        log_path_2 = os.path.join(os.getcwd(), log_folder, str_2+log_name+'_log.csv')
        
        #len of these entries is 2
        save_dict_3 = dict()
        save_dict_3['norm_cmt'] = self.norm_cmt
        save_dict_3['not_norm_cmt'] = self.not_norm_cmt
        str_3 = 'conf_matrix_'
        log_path_3 = os.path.join(os.getcwd(), log_folder, str_3+log_name+'_log.csv')
        
        #len of these entries is len(test_dset)
        save_dict_4 = dict()
        save_dict_4['eval_target'] = self.eval_tgt
        save_dict_4['eval_out_prob'] = self.eval_out_prob
        str_4 = 'eval_tgt_prob_'
        log_path_4 = os.path.join(os.getcwd(), log_folder, str_4+log_name+'_log.csv')
        
        df1 = pd.DataFrame.from_dict(save_dict_1)
        df1.fillna("-")
        df1.to_csv(log_path_1, index=False)
        
        df2 = pd.DataFrame.from_dict(save_dict_2)
        df2.to_csv(log_path_2, index=False)
        
        df3 = pd.DataFrame.from_dict(save_dict_3)
        df3.to_csv(log_path_3, index=False)
        
        df4 = pd.DataFrame.from_dict(save_dict_4)
        df4.to_csv(log_path_4, index=False)