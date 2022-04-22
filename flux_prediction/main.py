import os
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset

#import modules.dataset as ds
import modules.transformer as trf

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', 
                        help='Trains the transformer', 
                        action='store_true')
    parser.add_argument('--load', 
                        help='Loads a pretrained network', 
                        action='store_true')
    parser.add_argument('--curve-data', 
                        help='Creates and uses a curve training set', 
                        action='store_true')
    parser.add_argument('--sine-data', 
                        help='Creates and uses a sinusoidal training set', 
                        action='store_true')
    parser.add_argument('--data-sequence', 
                        help='Number and size of the sine sequences, default: 1000, 125', 
                        nargs=2)
    parser.add_argument('--flux-idx', 
                        help='Loads a flux dataset with given index, can be 1, 2, or 3. Default: None', 
                        nargs=1)
    parser.add_argument('--target-size', 
                        help='Size of target, default: 25', 
                        nargs=1)
    parser.add_argument('--model-dim', 
                        help='Dimention of the network, default: 64', 
                        nargs=1)
    parser.add_argument('--nheads', 
                        help='Number of heads, default: 1', 
                        nargs=1)
    parser.add_argument('--batch-size', 
                        help='Size of batch, default: 8', 
                        nargs=1)
    parser.add_argument('--epochs', 
                        help='Number of epochs, default: 100', 
                        nargs=1)
    parser.add_argument('--encoder', 
                        help='Which kind of positional encoder is used: t2v, pos(default)', 
                        nargs=1)
    args = parser.parse_args()

    TRAIN = args.train
    LOAD = args.load
    CURVE_DATA = args.curve_data
    SINE_DATA = args.sine_data
    
    if args.data_sequence == None:
        N_SEQUENCES = 1000
        SEQ_LENGTH = 125
    else:
        N_SEQUENCES = int(args.data_sequence[0])
        SEQ_LENGTH = int(args.data_sequence[1])
    
    if args.flux_idx == None:
        FLUX_IDX = args.flux_idx
    else:
        FLUX_IDX = int(args.flux_idx[0])
        
    if args.target_size == None:
        TARGET_SIZE = 25
    else:
        TARGET_SIZE = int(args.target_size[0])
        
    if args.model_dim == None:
        MODEL_DIM = 64
    else:
        MODEL_DIM = int(args.model_dim[0])
        
    if args.nheads == None:
        NHEADS = 1
    else:
        NHEADS = int(args.nheads[0])
        
    if args.batch_size == None:
        BATCH_SIZE = 8
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

    FLUX_DATA_DIR = './Magnetic_Flux_Area_Data'
    
    DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if SINE_DATA:
        transformer_save_file = os.path.join(os.getcwd(), 'sinusoid_transformer_params')
        eval_data = DataLoader(trf.SinusoidDataset(n_sequences=N_SEQUENCES, sequence_length=SEQ_LENGTH, target_size=TARGET_SIZE,
                                                   add_noise=False, 
                                                   #seed=14, 
                                                   device=DEV), batch_size=BATCH_SIZE)
    if CURVE_DATA:
        transformer_save_file = os.path.join(os.getcwd(), 'curve_transformer_params')
        eval_data = DataLoader(trf.CurveDataset(n_sequences=N_SEQUENCES, sequence_length=SEQ_LENGTH, target_size=TARGET_SIZE,
                                                add_noise=False, 
                                                #seed=14, 
                                                device=DEV), batch_size=BATCH_SIZE)
    if not SINE_DATA and not CURVE_DATA and FLUX_IDX is not None:
        transformer_save_file = os.path.join(os.getcwd(), 'flux_transformer_params')
        eval_data = DataLoader(trf.FluxDataset(FLUX_DATA_DIR, set_idx=FLUX_IDX, target_size=TARGET_SIZE, test=True, seed=12,
                                               device=DEV), batch_size=BATCH_SIZE)

    model = trf.FluxTransformer(data_dir=FLUX_DATA_DIR, set_idx=FLUX_IDX, target_size=TARGET_SIZE, model_d=MODEL_DIM,
                                nheads=NHEADS, encoding=ENCODER, epochs=EPOCHS, learning_rate=5e-3, gamma=0.98,
                                device=DEV).to(DEV)

    # Train or load the model
    if TRAIN:
        model.train_model(transformer_save_file, load_cp=True)
        if SINE_DATA:
            print('Model trained with sinusoidal dataset\nNumber of sequences: {}, length: {}'.format(N_SEQUENCES, SEQ_LENGTH))
        elif CURVE_DATA:
            print('Model trained with continuous curve dataset\nNumber of sequences: {}, length: {}'.format(N_SEQUENCES, SEQ_LENGTH))
        else:
            print('Model trained with flux dataset\nIndex: {}'.format(FLUX_IDX))
        print('Encoder: {}'.format(ENCODER))
    if LOAD:
        model.load_model(transformer_save_file)
        
    print('Device: {}'.format(DEV))
    print('Parameters used:')
    print('Epochs: {}\nTarget size: {}\nModel dimension: {}'.format(EPOCHS, TARGET_SIZE, MODEL_DIM))
    print('Heads: {}\nBatch size: {}'.format(NHEADS, BATCH_SIZE))

    # Plot some examples
    test_src, test_tgt = next(iter(eval_data))
    #model.show_example(test_src, test_tgt, 200)
    model.show_example(test_src, test_tgt, TARGET_SIZE*2)