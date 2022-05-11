###### import modules ######

import os
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset

import modules.transformer as trf
import modules.dataset as dst

import argparse

if __name__ == "__main__":
    
    ###### define parser ######
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', 
                        help='Trains the transformer', 
                        action='store_true')
    parser.add_argument('--load', 
                        help='Loads a pretrained network', 
                        action='store_true')
    parser.add_argument('--poly-data', 
                        help='Creates and uses a polynomial training set', 
                        action='store_true')
    parser.add_argument('--sine-data', 
                        help='Creates and uses a sinusoidal training set', 
                        action='store_true')
    parser.add_argument('--data-sequence', 
                        help='Number, size and target of function sequences, default: 1000, 100, 25', 
                        nargs=3)
    parser.add_argument('--flux-idx', 
                        help='Loads a flux dataset with given index, can be 1, 2, or 3. Default: None', 
                        nargs=1)
    parser.add_argument('--model-dim', 
                        help='Dimention of the network, default: 64', 
                        nargs=1)
    parser.add_argument('--nheads', 
                        help='Number of heads, default: 1', 
                        nargs=1)
    parser.add_argument('--batch-size', 
                        help='Size of batch, default: 4', 
                        nargs=1)
    parser.add_argument('--epochs', 
                        help='Number of epochs, default: 100', 
                        nargs=1)
    parser.add_argument('--encoder', 
                        help='Which kind of positional encoder is used: t2v, td, pos(default)', 
                        nargs=1)
    args = parser.parse_args()

    ###### parse arguments ######
    
    TRAIN = args.train
    LOAD = args.load
    POLY_DATA = args.poly_data
    SINE_DATA = args.sine_data
    
    if args.data_sequence == None:
        N_SEQUENCES = 1000
        SEQ_SOURCE_SIZE = 100
        SEQ_TARGET_SIZE = 25
    else:
        N_SEQUENCES = int(args.data_sequence[0])
        SEQ_SOURCE_SIZE = int(args.data_sequence[1])
        SEQ_TARGET_SIZE = int(args.data_sequence[2])
    
    if args.flux_idx == None:
        FLUX_IDX = args.flux_idx
    else:
        FLUX_IDX = int(args.flux_idx[0])
        
    if args.model_dim == None:
        MODEL_DIM = 64
    else:
        MODEL_DIM = int(args.model_dim[0])
        
    if args.nheads == None:
        NHEADS = 1
    else:
        NHEADS = int(args.nheads[0])
        
    if args.batch_size == None:
        BATCH_SIZE = 4
    else:
        BATCH_SIZE = int(args.batch_size[0])
        
    if args.epochs == None:
        EPOCHS = 100
    else:
        EPOCHS = int(args.epochs[0])

    if args.encoder == None:
        ENCODER = 'pos'
    else:
        ENCODER = str(args.encoder[0])

    FLUX_DATA_DIR = os.path.join(os.getcwd(), 'data', 'magnetic_flux_area_data')
    LSTM_DATA_DIR = os.path.join(os.getcwd(), 'data', 'lstm_data', 'multi_label_classification')
    
    DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    t_encoding_dim = 8
    e_layers = 3
    d_layers = 3
    
    ###### load dataset ######

    if SINE_DATA or POLY_DATA:
        FTR_SIZE = SEQ_TARGET_SIZE
        if SINE_DATA:
            DSET_TYPE = 'sin'
            transformer_save_file = os.path.join(os.getcwd(), 'model_params', 'sinusoid_transformer_params')
        if POLY_DATA:
            DSET_TYPE = 'pol'
            transformer_save_file = os.path.join(os.getcwd(), 'model_params', 'poly_transformer_params')
            
        train_dset, test_dset, eval_dset = dst.example_train_test_eval(DSET_TYPE, 
                                                                       n_sequences=N_SEQUENCES, source_size=SEQ_SOURCE_SIZE, 
                                                                       target_size=SEQ_TARGET_SIZE, 
                                                                       future_size=FTR_SIZE, add_noise=True, seed=42, device=DEV)
        eval_data = DataLoader(eval_dset, batch_size=BATCH_SIZE)
        TARGET_SIZE = SEQ_TARGET_SIZE
          
    if not SINE_DATA and not POLY_DATA and FLUX_IDX is not None:
        transformer_save_file = os.path.join(os.getcwd(), 'model_params', 'flux_transformer_params')
        train_dset, test_dset, eval_dset = dst.flux_train_test_eval(FLUX_DATA_DIR, FLUX_IDX, 
                                                                    seed=12, device=DEV)
        eval_data = DataLoader(eval_dset, batch_size=BATCH_SIZE)
        TARGET_SIZE = train_dset.get_target_size()
        
    ##### load the model ######
    
    model = trf.FluxTransformer(train_dset, test_dset,
                                model_d=MODEL_DIM, nheads=NHEADS, encoding=ENCODER,
                                time_encoding_dim=t_encoding_dim, enc_layers=e_layers, dec_layers=d_layers,
                                prediction_distance=25, epochs=EPOCHS, learning_rate=1e-3, gamma=0.97, device=DEV).to(DEV)

    ###### Train or load the model ######
    
    if TRAIN:
        model.train_model(transformer_save_file, load_cp=True)
        if SINE_DATA:
            print('Model trained with sinusoidal dataset\nNumber of sequences: {}, length: {}'.format(N_SEQUENCES, SEQ_SOURCE_SIZE))
        elif POLY_DATA:
            print('Model trained with polynomial dataset\nNumber of sequences: {}, length: {}'.format(N_SEQUENCES, SEQ_SOURCE_SIZE))
        else:
            print('Model trained with flux dataset\nIndex: {}'.format(FLUX_IDX))
        print('Encoder: {}'.format(ENCODER))
    if LOAD:
        model.load_model(transformer_save_file)
        
    print('Device: {}'.format(DEV))
    print('Parameters used:')
    print('Epochs: {}\nTarget size: {}\nModel dimension: {}'.format(EPOCHS, TARGET_SIZE, MODEL_DIM))
    print('Heads: {}\nBatch size: {}'.format(NHEADS, BATCH_SIZE))

    ###### Plot some examples ######
    
    test_src, test_tgt, test_ftr = next(iter(eval_data))
    model.show_example(test_src, test_tgt, test_ftr, plot_folder='plots')
    model.plot_loss(plot_folder='plots')