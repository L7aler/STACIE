###### import modules ######
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


###### load dataset ######

def mlabel_train_test_val(data_dir, n_labels=5, 
                           test_fraction=0.66, add_data=False, downsample=False,
                           seed=42, device=torch.device("cpu")):
    train_dset = MultiLabelDataset(data_dir,
                                   n_labels=n_labels,
                                   test_fraction=test_fraction,
                                   add_data=add_data, downsample=downsample,
                                   seed=seed,
                                   device=device)
    test_dset = MultiLabelDataset(data_dir,
                                  n_labels=n_labels,
                                  test_fraction=1.,
                                  seed=seed,
                                  test_set=True,
                                  device=device)
    val_dset = MultiLabelDataset(data_dir,
                                 n_labels=n_labels,
                                 test_fraction=1.,
                                 val_set=True, 
                                 seed=seed+1,
                                 device=device)
    return train_dset, test_dset, val_dset


###### multilabel dataset ######

class MultiLabelDataset(Dataset):
#class MultiLabelDataset(object):
    def __init__(self, data_dir, n_labels=5,
                 test_set=False, val_set=False, test_fraction=0.66, seed=42, 
                 add_data=False, downsample=False, random_split=True,
                 device=torch.device("cpu")):
        super(MultiLabelDataset, self).__init__()
        self.device = device

        # Load data.
        # Sources have shape [n_sequences, source_length=10, n_features=40]
        # Targets have shape [n_sequences, target_length=1, n_labels=5]
        
        self.len_data = []
        
        if test_set:
            sources = np.load(os.path.join(data_dir, 'test_input.npy'))
            targets = np.load(os.path.join(data_dir, 'test_target.npy'))
            self.len_data.append(f'loaded : {len(sources)}')
        elif val_set:
            sources = np.load(os.path.join(data_dir, 'val_input.npy'))
            targets = np.load(os.path.join(data_dir, 'val_target.npy'))
            self.len_data.append(f'loaded : {len(sources)}')
        else:
            sources = np.load(os.path.join(data_dir, 'train_input.npy'))
            targets = np.load(os.path.join(data_dir, 'train_target.npy'))
            self.len_data.append(f'loaded : {len(sources)}')
            if add_data:
                sources2 = np.load(os.path.join(data_dir, 'val_input.npy'))
                targets2 = np.load(os.path.join(data_dir, 'val_target.npy'))
                noflare_idx, flare_idx = self.split_flare(targets2)
                sources2 = sources2[flare_idx]
                targets2 = targets2[flare_idx]

                sources = np.concatenate((sources, sources2))
                targets = np.concatenate((targets, targets2))
                self.len_data.append(f'added data : {len(sources)}')
                
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
        if not val_set and not test_set:
            if downsample:
                downsampled_idx = self.downsample_data(targets)
                sources = sources[downsampled_idx]
                targets = targets[downsampled_idx]
                self.len_data.append(f'downsampled : {len(sources)}')
        if not val_set:         
            if random_split:
                test_size = int(test_fraction * len(sources))
                indexes = np.arange(len(sources))
                rnd_sample = test_sample = np.random.choice(indexes, test_size, replace=False)
                sources = sources[rnd_sample]
                targets = targets[rnd_sample]
                self.len_data.append(f'rnd split : {len(sources)}')
       
        self.source_data = sources
        self.target_data = targets
        
        #self.source_data = self.apply_mask(sources[test_sample])
        self.n_sequences = len(self.source_data)
        self.len_data.append(f'final : {self.n_sequences}')
        if downsample:
            self.class_weights = np.ones(self.n_labels)
        else:
            self.class_weights = self.calc_class_weights(self.target_data)
        
    def downsample_data(self, targets):
        class_idx = self.split_classes(targets, n_classes=self.n_labels)
        #print([len(label) for label in class_idx])
        min_class = min([len(label) for label in class_idx])
        out_idx = []
        for class_list in class_idx:
            if len(class_list) > min_class:
                idx_list = np.random.choice(class_list, min_class, replace=False)
                out_idx.extend(idx_list)
            else:
                out_idx.extend(class_list)
        return out_idx
        
    def split_flare(self, targets):
        noflare_idx = []
        flare_idx = []
        for i, tgt in enumerate(targets):
            class_idx = np.argmax(tgt[0])
            if class_idx == 0:
                noflare_idx.append(i)
            else:
                flare_idx.append(i)
        return noflare_idx, flare_idx
            
    def split_classes(self, targets, n_classes=5):
        index_list = [[] for _ in range(n_classes)]
        for i, tgt in enumerate(targets):
            class_idx = np.argmax(tgt[0])
            index_list[class_idx].append(i)
        #for class_num in index_list:
        #    print(len(class_num))
        return index_list
        
    def calc_class_weights(self, targets):
        class_num = np.zeros(self.n_labels)
        for tgt in targets:
            label_idx = np.argmax(tgt[0])
            class_num[label_idx] += 1
        class_num = class_num/self.n_sequences
        weights = np.ones(self.n_labels) - class_num
        return weights
                 
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
    
    def print_classes(self):
        class_idx = self.split_classes(self.target_data, n_classes=self.n_labels)
        total = 0
        for s in self.len_data:
            print(s)
        for i, label in enumerate(class_idx):
            num = len(label)
            total += num
            print(f'Class {i} : {num}')
        print(f'Total : {total}')
    
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
    
    def get_class_weights(self):
        return self.class_weights

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        src = self.source_data[idx]
        tgt = self.target_data[idx]

        src = torch.FloatTensor(src).to(self.device)
        tgt = torch.FloatTensor(tgt).to(self.device)
        return src, tgt