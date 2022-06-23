Tranformer-based flare prediction

1) unzip the data in the Flare_prediction/Data folder
2) before training BERT-mini, run savebert.py with the --labels N argument, N is the number of labels to classify
3) to train and test the model run main.py, use the --bert argument to train BERT-mini, --labels  and --epochs specify the number of classes and the number of epochs
4) the script outputs logs and plots in separate folders

post_processing.ipynb was used to compare different runs and to correct some plots. The plotting functions in main.py have been fixed and it's not necessary anymore to run this notebook to get the correct plots. 
