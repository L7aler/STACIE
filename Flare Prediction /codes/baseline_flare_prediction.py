from curses import init_pair
import xgboost as xgb
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report
import os
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import pickle
import matplotlib.pyplot as plt


def combine_class(data): #n_classes = 3
    if n_classes == 3:
        one_two = np.sum((data[:, 1], data[:, 2]), axis = 0)
        three_four = np.sum((data[:, 3], data[:, 4]), axis = 0)
        
        return np.vstack((data[:, 0], one_two, three_four)).T
    
    if n_classes == 2:
        no_flare = data[:, 0]
        flare = np.sum((data[:, 1], data[:, 2], data[:, 3], data[:, 4]), axis = 0)
        #flare = np.sum((data[:, -0]), axis = 0)
        
        return np.vstack((flare, no_flare)).T

def ma_f1(predt: np.ndarray, dMat: xgb.DMatrix):
    '''
    Log loss metric but weighted for class inbalance
    '''

    pred_lab = np.argmax(predt, axis = 1)
    
    y_true = dMat.get_label()
    
    loss = f1_score(y_true, pred_lab, average = 'macro', zero_division = 0)
    return 'ma_f1', float(loss)

def summarise_results(dMat, y, clf, best_iteration):
    pred = clf.predict(dMat, iteration_range = (0, best_iteration))
    pred_lab = np.argmax(pred, axis =1)
    y_true = np.argmax(y, axis = 1)
    print(classification_report(y_true, pred_lab))
    cm = confusion_matrix(y_true, pred_lab)
    
    if n_classes == 5:
        labels = ['No Flare', 'B', 'C',  'M', 'X']
    if n_classes == 3:
        labels = ['No Flare', 'C & B', 'M & X']
    if n_classes == 2:
        labels = ['No Flare', 'Flare']
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = np.array(labels))
    disp.plot()
    
def train_model(space):
    """
    Seperate function to train the model as will be used during hyperparameter tuning and re-training the final model.
    """
    
    if space['sample_weight'] == 'weighted':
        class_weights = class_weight.compute_sample_weight(
            class_weight='balanced',
            y=y_train)
        dtrain = xgb.DMatrix(X_train, label= np.argmax(y_train, axis = 1), weight = class_weights)
    else:
        dtrain = xgb.DMatrix(X_train, label = np.argmax(y_train, axis = 1))
    
    evallist = [(dtrain, 'train'), (dval, 'eval')] #early stopping uses the last in this list
    
    param = {
        'eta' : space['eta'],
        'max_depth' : int(space['max_depth']), 
        'gamma' : space['gamma'],
        'alpha' : space['alpha'],
        'min_child_weight' : int(space['min_child_weight']),
        'colsample_bytree' : space['colsample_bytree'],
        'subsample' : space['subsample'],
        'eval_metric' : str(space['loss']),
        'objective' : 'multi:softprob',
        'num_class' : n_classes
    }

    num_round = 50
    clf = xgb.train(param, dtrain, num_round, evals = evallist, verbose_eval = False, early_stopping_rounds = 10)
    
    return clf, clf.best_ntree_limit
     
    
def objective(space):
    """
    Objective function to do the hyperparameter tuning
    """
    model, best_iteration = train_model(space)
    f1 = ma_f1(model.predict(dval, iteration_range = (0, best_iteration)), dval)[1]
    print ("SCORE:", f1)
    return {'loss': -f1, 'status': STATUS_OK }
         

if __name__ == '__main__':
    input_dir = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1K6JU9H3NPWBhiahaDm81IHPEkDGoejnS/STACIE/flares/lstm_data/Multi_label_classification/'
    n_classes = 2
    
    #loading in the data
    X_train = np.load(input_dir + 'train_input.npy')[:, 0, :]
    X_val = np.load(input_dir + 'val_input.npy')[:, 0, :]
    X_test = np.load(input_dir + 'test_input.npy')[:, 0, :]
    
    y_train = np.load(input_dir + 'train_target.npy')[:, 0, :]
    y_val = np.load(input_dir + 'val_target.npy')[:, 0, :]
    y_test = np.load(input_dir + 'test_target.npy')[:, 0, :]
    
    if n_classes != 5:
        y_val = combine_class(y_val)
        y_train = combine_class(y_train)
        y_test = combine_class(y_test)

    dval = xgb.DMatrix(X_val, label= np.argmax(y_val, axis = 1))
    dtest = xgb.DMatrix(X_test, label= np.argmax(y_test, axis = 1))
    
    space={
        #boosting hps    
        'eta' : hp.uniform('eta', 0.1, 1.0),
        'loss' : hp.choice('loss', ['mlogloss', 'merror']), #eval metric not loss
        #tree hps
        'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'sample_weight' : hp.choice('sample_weight', ['weighted' , 'None']),
        #stochastic hps
        'subsample' : hp.uniform('subsample', 0.0, 1.0),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.1,1),
        #regularisation hps
        'gamma': hp.uniform ('gamma', 0, 0.4),
        'alpha' : hp.loguniform('alpha', 0, 4),
        #not hp
        'seed': 0
        }
    
    experiment_id = 'baseline_2_class_2'
    output_dir = '/Users/Max/Documents/MSc/2nd_Yr/adl/xgb/'
    max_evals = 20
    if os.path.isfile(output_dir + experiment_id + '.txt'):
        trials = pickle.load(open(output_dir + experiment_id + ".p", "rb"))
            
    else:
        trials = Trials()
        
    while len(trials.results) < max_evals:
        best_hyperparams = fmin(fn = objective,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = len(trials.results) + 10,
                            trials = trials)
        
        if len(trials.results) % 10 == 0:
            pickle.dump(trials, open(output_dir + experiment_id + ".p", "wb"))
    

    print("The best hyperparameters are : \n")
    print(best_hyperparams)
        
    best_hyperparams['loss'] = ['mlogloss', 'merror'][best_hyperparams['loss']]
    best_hyperparams['sample_weight'] = ['weighted', 'none'][best_hyperparams['sample_weight']]
    best_model, best_iteration = train_model(best_hyperparams)
    
    print("Validation Results")
    summarise_results(dval, y_val, best_model, best_iteration)
    print("Test Results:")
    summarise_results(dtest, y_test, best_model, best_iteration)
    
    plt.show()    
