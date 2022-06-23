# Solar Active Region (AR) property prediction

This directory contains scripts for downloading AR property data and for training neural networks to predict the evolution of these properties over time. The models are thus trained to solve a time-series-forecasting problem.

## Data

The AR property data from Joint Science Operations Center (JSOC) can be downloaded and processed with the command

`python SHARP_query.py`,

which will create a data directory with a training and validation set. Use the `SHARP_data_visualization.py` script to create plots that show some examples as well as the distribution of the different properties in the data.

## LSTM

In the 'code' directory, a notebook for training an LSTM model with this data is `LSTM_AR_network.ipynb`.

## Transformer

The code for training a Transformer-based model is in the `transformer_main.py` script. A few different Transformer-based architectures are given in `models.py`, which are imported in `transformer_main.py`. In addition, the `encoding.py` module contains several different data encoding algorithms (positional encoding, time2vec, time delay).