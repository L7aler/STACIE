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
    parser.add_argument('--add-data', 
                        help='Adds additional data from unused evaluation set (no flare series) to the training set', 
                        action='store_true')
    parser.add_argument('--downsample-data', 
                        help='Reduces the dataset based on the lowest class element count', 
                        action='store_true')
    parser.add_argument('--load', 
                        help='Loads a pretrained network and evaluates without training', 
                        action='store_true')
    parser.add_argument('--bert', 
                        help='Loads a pretrained bert network, for multilabel classification', 
                        action='store_true')
    parser.add_argument('--f1-loss', 
                        help='When classifying, changes the loss function from cross entropy to f1 score', 
                        action='store_true')
    parser.add_argument('--labels', 
                        help='Trains the transformer to predict flare types with given number of labels, values: 2, 3, 5(default)', 
                        nargs=1)
    parser.add_argument('--optim', 
                        help='Determines the optimizer: sgd, adamw, adam(default)', 
                        nargs=1)
    parser.add_argument('--enc-layers', 
                        help='Number of encoding layers, default: 4', 
                        nargs=1)
    parser.add_argument('--model-dim', 
                        help='Dimention of the network, default: 64', 
                        nargs=1)
    parser.add_argument('--nheads', 
                        help='Number of heads, default: 4', 
                        nargs=1)
    parser.add_argument('--batch-size', 
                        help='Size of batch, default: 16', 
                        nargs=1)
    parser.add_argument('--epochs', 
                        help='Number of epochs, default: 100', 
                        nargs=1)
    parser.add_argument('--encoder', 
                        help='Which kind of positional encoder is used: t2v, td, pos(default)', 
                        nargs=1)
    parser.add_argument('--out-name', 
                        help='Name of the output plot, default: None', 
                        nargs=1)
    args = parser.parse_args()

    ###### parse arguments ######
    
    ADD_DATA = args.add_data
    DOWNSAMPLE_DATA = args.downsample_data
    LOAD = args.load
    BERT = args.bert
    F1_LOSS = args.f1_loss
    
    if LOAD:
        TRAIN = False
    else:
        TRAIN = True
        
    if args.labels == None:
        NLABELS = 5
    else:
        NLABELS = int(args.labels[0])
        
    if args.optim == None:
        OPTIM = 'adam'
    else:
        OPTIM = str(args.optim[0])
        
    if args.model_dim == None:
        MODEL_DIM = 64
    else:
        MODEL_DIM = int(args.model_dim[0])
    if BERT:
        MODEL_DIM = 256
        
    if args.enc_layers == None:
        ENC_LAYERS = 4
    else:
        ENC_LAYERS = int(args.enc_layers[0])
        
    if args.nheads == None:
        NHEADS = 4
    else:
        NHEADS = int(args.nheads[0])
        
    if args.batch_size == None:
        BATCH_SIZE = 16
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
        
    if args.out_name == None:
        OUT_NAME = ''
    else:
        OUT_NAME = str(args.out_name[0])
        
    if F1_LOSS:
        loss_str = 'f1 score'
    else:
        loss_str = 'cross entropy'

    FLARE_DATA_DIR = os.path.join(os.getcwd(), 'data', 'lstm_data', 'multi_label_classification')
    DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    TIME_ENCODING_DIM = 1
    TARGET_SIZE = 1
        
    if OPTIM == 'adam' or OPTIM == 'adamw':
        lr = 2e-5
    elif OPTIM == 'sgd':
        lr = 0.2
    
    print('')
    
    if BERT:
        print('Training bert-mini for flare class prediction, with {} labels'.format(NLABELS))
    else:
        print('Training transformer for flare class prediction, with {} labels'.format(NLABELS))
        
    ###### load dataset ######

    transformer_save_file = os.path.join(os.getcwd(), 'model_params', f'mlabel_{NLABELS}_transformer_params')
    train_dset, test_dset, val_dset = dst.mlabel_train_test_val(FLARE_DATA_DIR, n_labels=NLABELS, seed=12, device=DEV,
                                                                test_fraction=1.,
                                                                add_data=ADD_DATA, downsample=DOWNSAMPLE_DATA)
    
    print('+++ Training dataset +++')
    train_dset.print_classes()
    print('')
    
    print('+++ Validation dataset +++')
    val_dset.print_classes()
    print('')
    
    print('+++ Test dataset +++')
    test_dset.print_classes()
    print('')
    
    print('+++ Model parameters +++')
    
    if BERT:
        print(f'Dimension : {MODEL_DIM}')
    else:
        print(f'Heads : {NHEADS}\nDimension : {MODEL_DIM}\nEncoding layers : {ENC_LAYERS}')
    
    print(f'Encoder : {ENCODER}\nLoss function : {loss_str}\nOptimizer : {OPTIM}\nLearning rate : {lr}')
    print(f'Batch size : {BATCH_SIZE}\nEpochs : {EPOCHS}\nDevice : {DEV}\nOutput name : {OUT_NAME}') 
        
        
    ##### load the model ######
    
    model = trf.FlareClassificationTransformer(train_dset, val_dset, bert=BERT, 
                                f1_loss=F1_LOSS, optimizer=OPTIM, model_dim=MODEL_DIM, nheads=NHEADS, encoding=ENCODER,
                                time_encoding_dim=TIME_ENCODING_DIM, enc_layers=ENC_LAYERS, batch_size=BATCH_SIZE,
                                epochs=EPOCHS, learning_rate=lr, gamma=0.97, device=DEV).to(DEV)

    
    ###### Train or load the model ######
    
    print('')
    if TRAIN:
        print('+++ Training model +++')
        print('')
        model.train_model(transformer_save_file)
        
    if LOAD:
        if BERT:
            pass
        else:
            print('+++ Loading model +++')
            model.load_model(transformer_save_file)
    print('')

    ###### Plot some examples ######
    print('+++ Model evaluation +++')
    test_data = DataLoader(test_dset, batch_size=BATCH_SIZE)
    model.plot_confusion_matrix(test_data, plot_folder='plots', plot_name=OUT_NAME, show=False)
    model.plot_loss(plot_folder='plots', plot_name=OUT_NAME+'_classifier_loss.png', show=False)
    model.save_logs(log_folder='logs', log_name=OUT_NAME)