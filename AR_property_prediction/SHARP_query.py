import drms
import os
import spectres
import numpy as np
import pandas as pd
from tqdm import tqdm

starting_year = 2010
interval = 1
months = np.arange(12 * 12, step=interval)  # up to 2022

# Period strings for query
periods = [f'{starting_year + month // 12}.{month % 12 + 1:02d}.01 - '
           f'{starting_year + (month + interval) // 12}.{(month + interval) % 12 + 1:02d}.01' for month in months]

results_dir = './sharp_queries'
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

download_data = True
make_train_test = True

sequence_length = 300
resample_interval = 5  # use odd numbers only
shift = 20
test_fraction = 0.3

# ======================================================
# DOWNLOADING DATA
# ======================================================

c = drms.Client()
c.series(r'hmi\.sharp_')
si = c.info('hmi.sharp_cea_720s')

# All features
# features = 'T_REC, HARPNUM, TOTUSJH, TOTBSQ, TOTPOT, TOTUSJZ, ABSNJZH, SAVNCPP,' \
#            'CRLT_OBS, CRLN_OBS, USFLUX, AREA_ACR, MEANPOT, R_VALUE, SHRGT45, MEANSHR,' \
#            'MEANGAM, MEANGBT, MEANGBZ, MEANGBH, MEANJZH, MEANJZD, ' \
#            'MEANALP, TOTFX, TOTFY, TOTFZ, EPSX, EPSY, EPSZ, Bdec, Cdec, Mdec,' \
#            'Xdec, Edec, logEdec, Bhis, Chis, Mhis, Xhis, Bhis1d, Chis1d, Mhis1d,' \
#            'Xhis1d, Xmax1d'

features = 'USFLUX, AREA_ACR, TOTUSJH, TOTPOT, TOTUSJZ, ABSNJZH, SAVNCPP, MEANPOT, SHRGT45,' \
           'MEANGAM, MEANGBT, MEANGBZ, MEANGBH, MEANJZH, MEANJZD, MEANALP'

pbar = tqdm(periods, total=len(periods), desc='Downloading SHARP data')
pbar.set_postfix({'Data points per month': np.nan})

# Download active region data for every month between 2010 and 2022
for period in tqdm(periods, total=len(periods), desc='Downloading SHARP data'):
    query_string = f'hmi.sharp_cea_720s[][{period}][? (CRLN_OBS < 360) AND (CRLT_OBS < 360) AND (USFLUX > 4e21) ?]'
    data = c.query(query_string, key=features)
    data.to_csv(os.path.join(results_dir, period + '.csv'), index='False')
    pbar.update(1)
    pbar.set_postfix({'Data points per month': len(data)})

# ======================================================
# NORMALIZING DATA
# ======================================================

# Concatenate the data from every month
dfs = []
for period in periods:
    data_file = os.path.join(results_dir, period + '.csv')
    period_df = pd.read_csv(data_file, index_col=0)
    dfs.append(period_df)

all_data = pd.concat(dfs).dropna()
all_data = all_data.set_index([pd.Index(np.arange(0, len(all_data), 1))])

# HARPNUM is the active region identifier
harp_nums = all_data['HARPNUM'].unique()
print("Number of different AR's in the data:", len(harp_nums))

# Divide the active regions among the train and test set
np.random.seed(42)
test_harp_nums = np.random.choice(harp_nums, int(len(harp_nums) * test_fraction), replace=False)
train_harp_nums = np.array(list(set(harp_nums) - set(test_harp_nums)))

# Construct one or more data sequences for every active region with sufficient data
sequences = [[], []]  # train and test
labels = ['train', 'test']

for i, harp_num_set in enumerate([train_harp_nums, test_harp_nums]):
    pbar = tqdm(harp_num_set, total=len(harp_num_set), desc=f"Creating {labels[i]} sequences")
    tot_sequences = 0
    for harp_num in pbar:
        ar_indices = np.where(all_data['HARPNUM'] == harp_num)[0]
        single_ar_data = all_data.iloc[ar_indices]
        n_data_points = len(single_ar_data)

        start = 0
        n_sequences = 0

        while start + sequence_length < n_data_points:
            sequence = single_ar_data[start:start + sequence_length].to_numpy().astype(float)[None, ...]
            sequences[i].append(sequence)
            start += shift
            n_sequences += 1

        tot_sequences += n_sequences
        pbar.set_postfix({'tot_sequences': tot_sequences})

# Create a train and test dataset with resampled and normalized features

set_dir = 'set_5'  # dataset identifier
set_path = os.path.join('./data', set_dir)

if not os.path.exists(set_path):
    os.mkdir(set_path)

features = all_data.columns.to_list()
n_features = len(all_data.columns.to_list())

for i, set_type in enumerate(labels):

    # Select the train or test sequences
    data = np.concatenate(sequences[i])
    save_file = os.path.join(set_path, f'{set_type}_normalized')

    # Resample data to reduce the number of points
    new_wavs = np.arange(resample_interval // 2, sequence_length, resample_interval)
    spec_wavs = np.arange(0, sequence_length, 1)
    resampled_data = np.zeros(shape=(len(data), int(sequence_length / resample_interval), n_features))
    pbar = tqdm(enumerate(data), total=len(data), desc=f'Resamling {set_type} features')
    for sequence_idx, d in pbar:
        for feature_idx in range(n_features):
            resampled_data[sequence_idx, :, feature_idx] = spectres.spectres(new_wavs, spec_wavs, d[:, feature_idx],
                                                                             spec_errs=None, fill=None, verbose=True)

    # Normalize the data
    normalized_data = []
    for feature_idx in range(data.shape[-1]):
        feature_vals = resampled_data[:, :, feature_idx]

        # Take the log of features that span multiple orders of magnitudes
        if features[feature_idx] in ['USFLUX', 'AREA_ACR', 'TOTUSJH', 'TOTPOT', 'TOTUSJZ',
                                     'ABSNJZH', 'SAVNCPP', 'MEANPOT', 'SHRGT45']:
            # Take the lowest value that is not 0
            min_val = np.min(np.where(feature_vals > 0, feature_vals, np.max(feature_vals)))
            # Replace zeros with that value
            feature_vals = np.where(feature_vals == 0., min_val, feature_vals)
            feature_vals = np.log10(feature_vals)

        # Standardization
        feature_vals = (feature_vals - np.mean(feature_vals, axis=(0, 1))) / np.std(feature_vals, axis=(0, 1))

        normalized_data.append(feature_vals[..., None])

    # Randomize the order of the sequences
    _sequences = np.random.permutation(np.concatenate(normalized_data, axis=-1))

    print('Sequence data shape:', _sequences.shape)
    print('Number of data points:', _sequences.shape[0] * _sequences.shape[1])
    print('Example', _sequences[0, 0])

    # Save the normalized data
    np.save(save_file, _sequences)
