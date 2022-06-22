import os
import numpy as np
import matplotlib.pyplot as plt

features = ['USFLUX', 'AREA_ACR', 'TOTUSJH', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'MEANPOT', 'SHRGT45',
            'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'MEANJZD', 'MEANALP']
set_dir = 'set_5'
set_path = os.path.join('./AR_property_prediction_data', set_dir)
source_size = 40

for i, set_type in enumerate(['train', 'test']):
    data = np.load(os.path.join(set_path, f'{set_type}_normalized.npy'))

    # Plot feature example
    print('Data shape', data.shape)
    _, ax = plt.subplots(4, 4, figsize=(16, 16))
    for w in range(4):
        for h in range(4):
            for seq in range(0, 10):
                ax[w, h].plot(data[seq, :, 4 * w + h])
            ax[w, h].set_title(f'feature {4 * w + h} ({features[4 * w + h]})')
    plt.savefig(os.path.join(set_path, f'{set_type}_examples.png'))
    plt.show()
    plt.close()

    # Plot features distribution
    fig, ax = plt.subplots(4, 4, figsize=(16, 16))
    for w in range(4):
        for h in range(4):
            feature_vals = data[:, :, 4 * w + h]
            ax[w, h].hist(feature_vals[:, :source_size].reshape(-1), bins=100, histtype='step', density=True,
                          label='source')
            ax[w, h].hist(feature_vals[:, source_size:].reshape(-1), bins=100, histtype='step', density=True,
                          label='target')
            ax[w, h].set_yscale('log')
            ax[w, h].set_title(f'feature {4 * w + h} ({features[4 * w + h]})')
            ax[w, h].legend()
    plt.savefig(os.path.join(set_path, f'{set_type}_data_distribution.png'))
    plt.show()
    plt.close()
