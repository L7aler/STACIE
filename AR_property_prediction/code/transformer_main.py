import os
import torch
from torch.utils.data import DataLoader
from transformer import SinusoidDataset, PolynomialDataset, FluxDataset
from transformer import FluxTransformer

if __name__ == "__main__":
    flux_data_dir = '../data'
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Running on device:', dev)

    model_dim = 128
    nhead = 8
    e_layers = 3
    d_layers = 3

    ep = 50
    bat_size = 64
    learning_rate = 1e-3
    gamma = 0.96
    baseline = False

    set_id = '5'  # ['5', 'sin', 'pol']
    model_identity = 'transformer_encoder'  # ['transformer_encoder', 'full_transformer', 'pretrained_bert', 'lstm']

    encoding_type = 'td'  # ['pos', 't2v', 'td']
    t_encoding_dim = 8  # used if encoding type is t2v or td

    if model_identity == 'pretrained_bert':
        model_dim = 256
        src_size = 35
        tgt_size = 5
        ftr_size = 20
    elif model_identity == 'full_transformer':
        src_size = 35
        tgt_size = 5
        ftr_size = 20
    else:
        src_size = 40
        tgt_size = 20
        ftr_size = 0

    print('Model:', model_identity)

    transformer_save_file = os.path.join(os.getcwd(), f'{model_identity}_params')

    model = FluxTransformer(data_dir=flux_data_dir, set_id=set_id, model_id=model_identity,
                            source_size=src_size, target_size=tgt_size, future_size=ftr_size,
                            model_d=model_dim, nheads=nhead, encoding=encoding_type,
                            time_encoding_dim=t_encoding_dim, enc_layers=e_layers, dec_layers=d_layers,
                            epochs=ep, batch_size=bat_size, learning_rate=learning_rate, gamma=gamma,
                            use_baseline=baseline, device=dev).to(dev)

    model.train_model(transformer_save_file, load_cp=False)

    if set_id == 'sin':
        val_data = DataLoader(SinusoidDataset(n_sequences=100, source_size=src_size, target_size=tgt_size,
                                              future_size=ftr_size, add_noise=True, seed=14, device=dev),
                              batch_size=9)
    elif set_id == 'pol':
        val_data = DataLoader(PolynomialDataset(n_sequences=100, source_size=src_size, target_size=tgt_size,
                                                future_size=ftr_size, add_noise=True, seed=14, device=dev),
                              batch_size=9)
    else:
        val_data = DataLoader(FluxDataset(flux_data_dir, set_idx=set_id, source_size=src_size, target_size=tgt_size,
                                          future_size=ftr_size, validation=True, device=dev),
                              batch_size=bat_size)

    val_src, val_tgt, val_ftr = next(iter(val_data))
    model.show_example(val_src, val_tgt, val_ftr)
