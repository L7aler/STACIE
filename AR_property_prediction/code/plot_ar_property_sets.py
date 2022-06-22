import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformer import FluxTransformer

flux_data_dir = '../data'
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Running on device:', dev)

src_size = 40
tgt_size = 20
ftr_size = 0

model_dim = 128
nhead = 8
t_encoding_dim = 8
e_layers = 3
d_layers = 3

ep = 50
bat_size = 64
learning_rate = 1e-4
gamma = 0.98
baseline = True

set_id = '5'
model_identity = 'transformer_encoder'  # ['transformer_encoder', 'full_transformer', 'pretrained_bert', 'lstm']
encoding_type = 'td'  # ['pos', 't2v', 'td']

model = FluxTransformer(data_dir=flux_data_dir, set_id=set_id, model_id=model_identity,
                        source_size=src_size, target_size=tgt_size, future_size=ftr_size,
                        model_d=model_dim, nheads=nhead, encoding=encoding_type,
                        time_encoding_dim=t_encoding_dim, enc_layers=e_layers, dec_layers=d_layers,
                        epochs=ep, batch_size=bat_size, learning_rate=learning_rate, gamma=gamma,
                        use_baseline=baseline, device=dev).to(dev)

transformer_save_file = os.path.join(os.getcwd(), f'{model_identity}_params')
model.load_model(transformer_save_file)

# features = ['USFLUX', 'AREA_ACR', 'TOTUSJH', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'MEANPOT', 'SHRGT45',
#             'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'MEANJZD', 'MEANALP']

selected_features = np.array([0, 1, 2, 3, 4, 7, 9, 10, 11, 12])
# features = np.array(features)[selected_features]

plt.rcParams["font.family"] = ["Times New Roman"]

plot_indices = [[10547, 201, 2200], [2244, 2206, 10531], [10504, 2224, 10513]]

data_file = os.path.join(flux_data_dir, 'set_5', 'test_normalized.npy')
sequences = np.load(data_file, allow_pickle=True)[:, :, selected_features]

model.eval()

fig, ax = plt.subplots(3, 3, figsize=(16, 12))

for i in range(3):
    for j in range(3):
        seq_idx = plot_indices[i][j]
        src = torch.FloatTensor(sequences[seq_idx, :40, :][None, ...]).to(dev)
        tgt = torch.FloatTensor(sequences[seq_idx, 40:, :][None, ...]).to(dev)

        # Prediction of the target
        pred_tgt = model(src, tgt)

        # Source timesteps
        src_t = np.arange(40)

        # Target timesteps
        tgt_t = np.arange(39, 60)

        # Concatenate last step of target for smooth plot
        tgt = torch.cat((src[:, -1:, :], tgt), dim=1)
        pred_tgt = torch.cat((src[:, -1:, :1], pred_tgt), dim=1)

        ax[i, j].plot(src_t, src[0, :, 0].cpu().detach().numpy(), c='black', label='Source')
        ax[i, j].plot(tgt_t, tgt[0, :, 0].cpu().detach().numpy(), '--', c='black', label='Target')
        ax[i, j].plot(tgt_t, pred_tgt[0, :, 0].cpu().detach().numpy(), '--', c='red', label='Prediction')
        ax[i, j].set_ylabel('Magnetic Flux (Mx)', fontsize=12)
        ax[i, j].set_title(f'Set {seq_idx}')
        ax[i, j].legend(frameon=False, fontsize=12)
        ax[i, j].set_xlabel(r'Time ($\times 60$min)', fontsize=12)

        ax[i, j].minorticks_on()
        ax[i, j].tick_params(which='both', bottom=True, top=True, left=True, right=True)
        ax[i, j].tick_params(which='major', length=8, direction='in', labelsize=10)
        ax[i, j].tick_params(which='minor', length=4, direction='in', labelsize=10)
plt.tight_layout()
plt.show()