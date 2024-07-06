import sys, os; sys.path.append( '/kaggle/input/smilesx-demo/SMILES-X/')

# Load packages

import math
import logging
import datetime
import pickle as pkl

from scipy.ndimage.interpolation import shift
from sklearn.preprocessing import RobustScaler

from rdkit import Chem
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl


import pandas as pd
from importlib import reload

from glob import glob
import math
import time
import logging
import datetime
import collections
import pickle as pkl

import numpy as np
import pandas as pd
from tabulate import tabulate
from typing import List, Optional

import matplotlib.pyplot as plt

import logging
import pandas as pd
import numpy as np

from typing import Optional
from typing import List

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix



import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
from tensorflow.keras import backend as K

from sklearn.model_selection import GroupKFold, StratifiedKFold

from SMILESX import utils, token, augm
from SMILESX import model, bayopt, geomopt
from SMILESX import visutils, trainutils
from SMILESX import loadmodel

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder

import duckdb
import pandas as pd


import os
import glob
from pickle import load

import numpy as np
import pandas as pd

from typing import Optional
from typing import List

from rdkit import Chem

# Suppress Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
# Ignore shape mismatch warnings related to the attention layer
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.models import Model, load_model
from tensorflow.keras import metrics
from tensorflow.keras import backend as K

from SMILESX import utils, model, token, augm


np.random.seed(seed=123)
np.set_printoptions(precision=3)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # was '3'   # Suppress Tensorflow warnings
tf.autograph.set_verbosity(3)
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
logger = logging.getLogger()
logging.getLogger("tensorflow").setLevel(logging.ERROR)



if ('df1' in globals())==False:
    def get_df( N=N, O=O ):
        con = duckdb.connect()
        
        if DEBUG:            
            df = con.query(f"""(SELECT *
                                    FROM parquet_scan('{train_path}')
                                    WHERE (binds = 0) 
                                    ORDER BY random()
                                    LIMIT {N})
                                    UNION ALL
                                    (SELECT *
                                    FROM parquet_scan('{train_path}')
                                    WHERE (binds = 1) 
                                    ORDER BY random()
                                    LIMIT {N})""").df()
        else: # 
            df = con.query(f"""(SELECT *
                                    FROM parquet_scan('{train_path}')
                                    WHERE (binds = 0) AND (protein_name = 'BRD4')   
                                    ORDER BY random()
                                    LIMIT {N})
                                    UNION ALL
                                    (SELECT *
                                    FROM parquet_scan('{train_path}')
                                    WHERE (binds = 1) AND (protein_name = 'BRD4') 
                                    ORDER BY random()
                                    LIMIT {N})""").df()
            df_h = con.query(f"""(SELECT *
                                    FROM parquet_scan('{train_path}')
                                    WHERE (binds = 0) AND (protein_name = 'HSA')
                                    ORDER BY random()
                                    LIMIT {N})
                                    UNION ALL
                                    (SELECT *
                                    FROM parquet_scan('{train_path}')
                                    WHERE (binds = 1) AND (protein_name = 'HSA')
                                    ORDER BY random()
                                    LIMIT {N})""").df()
            df_s = con.query(f"""(SELECT *
                                    FROM parquet_scan('{train_path}')
                                    WHERE (binds = 0) AND (protein_name = 'sEH')
                                    ORDER BY random()
                                    LIMIT {N})
                                    UNION ALL
                                    (SELECT *
                                    FROM parquet_scan('{train_path}')
                                    WHERE (binds = 1) AND (protein_name = 'sEH')
                                    ORDER BY random()
                                    LIMIT {N})""").df()
        con.close()

        try:
            df1 = pd.concat( [df, df_s, df_h ])
        except:
            df1 = df
  
        df1['protein_code1'] = (df1['protein_name'] == 'BRD4' ).astype(np.int8)
        df1['protein_code2'] = (df1['protein_name'] == 'HSA' ).astype(np.int8)
        df1['protein_code3'] = (df1['protein_name'] == 'sEH' ).astype(np.int8)

        return df1
    
    df1 = get_df( N=N, O=O )



def generate_ecfp(molecule, radius=3, bits=1024*2):
    if molecule is None:
        return None
    return list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits))
    
field_names = ['protein_code1','protein_code2','protein_code3', 'molecule_smiles','buildingblock1_smiles','buildingblock2_smiles', 'buildingblock3_smiles'] 
y = df1[['binds']] #.tolist()

# Split the development set into train and validation subsets
Y,X = {},{}
X['trn'], X['val'], Y['trn'], Y['val'] = train_test_split( df1[ field_names ], y, test_size=0.2, random_state=42)
Y['trn'].head()


for t in ['trn', 'val']:
    Y[t]=Y[t].astype( np.int8 )



logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog('rdApp.error')

np.set_printoptions(precision=3)

class StopExecution(Exception):
    """Clean execution termination (no warnings).
    """
    def _render_traceback_(self):
        pass

def set_gpuoptions(n_gpus = 1,
                   gpus_list = None,
                   gpus_debug = False,
                   print_fn=logging.info):    
    # To find out which devices your operations and tensors are assigned to
    tf.debugging.set_log_device_placement(gpus_debug)
    if gpus_list is not None:
        gpu_ids = [int(iid) for iid in gpus_list]
    elif n_gpus>0:
        gpu_ids = [int(iid) for iid in range(n_gpus)]
    else:
        print_fn("Number of GPUs to be used is set to 0. Proceed with CPU.")
        print_fn("")
        device = "/cpu:0"
        strategy = tf.distribute.OneDeviceStrategy(device=device)
        devices = tf.config.list_logical_devices('CPU')
        return strategy, devices
        
    gpus = tf.config.experimental.list_physical_devices('GPU')    
    if gpus:
        if 1:
            #try:
            # Keep only requested GPUs
            gpus = [gpus[i] for i in gpu_ids]
            # Currently, memory growth needs to be the same across GPUs
            # for gpu in gpus:
            #     tf.config.experimental.set_memory_growth(gpu, False )
            tf.config.experimental.set_visible_devices(gpus, 'GPU')
            devices = tf.config.list_logical_devices('GPU')
            print_fn("{} Physical GPU(s), {} Logical GPU(s) detected and configured.".format(len(gpus), len(devices)))
        #except RuntimeError as e: 
        #    print_fn(e)
                
        gpus_list_len = len(devices)
        if gpus_list_len > 0:
            if gpus_list_len > 1: 
                strategy = tf.distribute.MirroredStrategy()
            else:
                # Important! The command list_logical_devices renumerates the gpus starting from 0
                # The number here will be 0 regardless the requested GPU number
                device = "/gpu:0"
                strategy = tf.distribute.OneDeviceStrategy(device=device)
            print_fn('{} GPU device(s) will be used.'.format(strategy.num_replicas_in_sync))
            print_fn("")
            return strategy, devices
    else:
        device = "/cpu:0"
        strategy = tf.distribute.OneDeviceStrategy(device=device)
        devices = tf.config.list_logical_devices('CPU')
        print_fn("No GPU is detected in the system. Proceed with CPU.")
        print_fn("")
        return strategy, devices


def log_setup(save_dir, name, verbose):
    # Setting up logging
    currentDT = datetime.datetime.now()
    strDT = currentDT.strftime("%Y-%m-%d_%H:%M:%S")
       
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)    
    formatter = logging.Formatter(fmt='%(asctime)s:   %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    
    # Remove existing handlers if any
    logger.handlers.clear()
    
    # Logging to the file
    logfile = '{}/{}_{}.log'.format(save_dir, name, strDT)
    handler_file = logging.FileHandler(filename=logfile, mode='w')
    handler_file.setLevel(logging.INFO)
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    
    # Logging to console
    if verbose:
        handler_stdout = logging.StreamHandler(sys.stdout)
        handler_stdout.setLevel(logging.INFO)
        handler_stdout.setFormatter(formatter)
        logger.addHandler(handler_stdout)
    
    return logger, logfile

def rand_split(smiles_input, prop_input, extra_input, err_input, train_val_idx, test_idx, bayopt = False):    
    # Assure random training/validation split (test set is unchanged)
    np.random.seed(42)
    np.random.shuffle(train_val_idx)
    
    # How many samples goes to training
    # We perform 7:2:1 split for train:val:test sets
    train_smpls = math.ceil(train_val_idx.shape[0]*6/9)
    
    train_idx = train_val_idx[:train_smpls]
    valid_idx = train_val_idx[train_smpls:]
    
    x_train = smiles_input[train_idx]
    y_train = prop_input[train_idx]
    extra_train = extra_input[train_idx] if extra_input is not None else None
    
    x_valid = smiles_input[valid_idx]
    y_valid = prop_input[valid_idx]
    extra_valid = extra_input[valid_idx] if extra_input is not None else None

    if bayopt:
        # No need of test set for Bayesian optimisation
        return x_train, x_valid, extra_train, extra_valid, y_train, y_valid
    
    x_test = smiles_input[test_idx]
    y_test = prop_input[test_idx]
    extra_test = extra_input[test_idx] if extra_input is not None else None
    
    # Only split when errors are provided
    err_test = err_input[test_idx] if err_input is not None else None
    err_train = err_input[train_idx] if err_input is not None else None
    err_valid = err_input[valid_idx] if err_input is not None else None

    logger.info("Train/valid/test splits: {0:0.2f}/{1:0.2f}/{2:0.2f}".format(\
                                          x_train.shape[0]/smiles_input.shape[0],\
                                          x_valid.shape[0]/smiles_input.shape[0],\
                                          x_test.shape[0]/smiles_input.shape[0]))
    logger.info("")
    
    return x_train, x_valid, x_test, extra_train, extra_valid, extra_test, y_train, y_valid, y_test, err_train, err_valid, err_test


def robust_scaler(train, valid, test, file_name, ifold):    
    if ifold is not None:
        scaler_file = '{}_Fold_{}.pkl'.format(file_name, ifold)
        try:
            # If the scaler exists, load and make no changes
            scaler = pkl.load(open(scaler_file, "rb"))
        except (OSError, IOError) as e:
            # If doens't exist, create and fit to training data
            scaler = RobustScaler(with_centering=True, 
                                  with_scaling=True, 
                                  quantile_range=(5.0, 95.0), 
                                  copy=True)
            scaler_fit = scaler.fit(train)
            # Save scaler for future usage (e.g. inference)
            pkl.dump(scaler, open(scaler_file, "wb"))
            logger = logging.getLogger()
            logger.info("Scaler: {}".format(scaler_fit))
    else: # The scalers are not saved during Bayesian optimization
        scaler = RobustScaler(with_centering=True, 
                              with_scaling=True, 
                              quantile_range=(5.0, 95.0), 
                              copy=True)
        scaler_fit = scaler.fit(train)
    
    train_scaled = scaler.transform(train)
    valid_scaled = scaler.transform(valid)
    if test is not None:
        test_scaled = scaler.transform(test)
    else:
        test_scaled = None
    
    return train_scaled, valid_scaled, test_scaled, scaler

def smiles_concat(smiles_list):
    """ Concatenate multiple SMILES in one via 'j'
    
    Parameters
    ----------
    smiles_list: array
        Array of SMILES to be concatenated along axis=0 to form a single SMILES.
    
    Returns
    -------
    concat_smiles_list
        List of concatenated SMILES, one per data point.
    """
    concat_smiles_list = []
    for smiles in smiles_list:
        concat_smiles_list.append('j'.join([ismiles for ismiles in smiles if ismiles != '']))
    return concat_smiles_list

def mean_result(smiles_enum_card, preds_enum):
    """Compute mean and median of predictions
    
    Parameters
    ----------
    smiles_enum_card: list(int)
        List of indices that are the same for the augmented SMILES originating from the same original SMILES
    preds_enum: np.array
        Predictions for every augmented SMILES for every predictive model

    Returns
    -------
        preds_mean: float
            Mean over predictions augmentations and models
        preds_std: float
            Standard deviation over predictions augmentations and models
    """
    
    preds_ind = pd.DataFrame(preds_enum, index = smiles_enum_card)
    preds_mean = preds_ind.groupby(preds_ind.index).apply(lambda x: np.mean(x.values)).values.flatten()
    preds_std = preds_ind.groupby(preds_ind.index).apply(lambda x: np.std(x.values)).values.flatten()

    return preds_mean, preds_std

# Learning curve plotting
def learning_curve(train_loss, val_loss, save_dir: str, data_name: str, ifold: int, run: int) -> None:

    fig = plt.figure(figsize=(6.75, 5), dpi=200)
    ax = fig.add_subplot(111)

    ax.set_ylim(0, max(max(train_loss), max(val_loss))+0.005)

    plt.ylabel('Loss (RMSE, scaled)', fontsize=18)
    plt.xlabel('Epoch', fontsize=18)
    
    ax.plot(train_loss, color='#3783ad')
    ax.plot(val_loss, color='#a3cee6')

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis="x", direction="inout")
    ax.tick_params(axis="y", direction="inout")

    # Ticks decoration
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(axis="x",
                   which="minor",
                   direction="out",
                   top=True,
                   labeltop=True,
                   bottom=True,
                   labelbottom=True)

    ax.tick_params(axis="y",
                   which="minor",
                   direction="out",
                   right=True,
                   labelright=True,
                   left=True,
                   labelleft=True)
    ax.legend(['Train', 'Validation'], loc='upper right', fontsize=14)
    plt.savefig('{}/{}_LearningCurve_Fold_{}_Run_{}.png'\
                .format(save_dir, data_name, ifold, run), bbox_inches='tight')
    plt.close()
    
from keras import backend as K
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

## Compute diverse scores to quantify model's performance on classification tasks
def classification_metrics(y_true, y_pred):

    # Compute the average class predictions for binary classification
    y_pred_class = (y_pred > 0.5).astype("int8")

    y_true = y_true.astype("int8")
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_true, y_pred_class)
    # precision tp / (tp + fp)
    precision = precision_score(y_true, y_pred_class)
    # recall: tp / (tp + fn)
    recall = recall_score(y_true, y_pred_class)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_true, y_pred_class)
    # AUC
    prp_precision, prp_recall, _ = precision_recall_curve(y_true, y_pred)
    prp_auc = auc(prp_recall, prp_precision)
    # confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred_class)
    
    return accuracy, precision, recall, f1, prp_auc, conf_mat

def negate_acc(y_true, y_pred):

    # Compute the average class predictions for binary classification
    y_pred_class = tf.cast( y_pred, tf.int8 ) 

    y_true = tf.cast( y_true, tf.int8 )
    
    # precision tp / (tp + fp)
    precision = precision_score(y_true, y_pred_class)
    
    # recall: tp / (tp + fn)
    recall = recall_score(y_true, y_pred_class)

    if 0:
        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(y_true, y_pred_class)    
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(y_true, y_pred_class)
        # AUC
        prp_precision, prp_recall, _ = precision_recall_curve(y_true, y_pred)
        prp_auc = auc(prp_recall, prp_precision)
        
        # confusion matrix
        conf_mat = confusion_matrix(y_true, y_pred_class)
    
    return 1 - (precision + recall)/2
    


def print_stats(trues, preds, errs_pred=None, prec: int = 4, model_type = 'regression'):     
    set_names = ['test', 'validation', 'train']

    if errs_pred is None:
        errs_pred = [None]*len(preds)

    outputs = []
    for true, pred, err_pred in zip(trues, preds, errs_pred):
        true, pred = np.array(true).ravel(), np.array(pred).ravel()

        if model_type == 'regression':
            rmse = np.sqrt(mean_squared_error(true, pred))
            mae = mean_absolute_error(true, pred)
            r2 = r2_score(true, pred)

            prec_rmse = output_prec(rmse, prec)
            prec_mae = output_prec(mae, prec)

            if err_pred is None:                
                logging.info('Model performance metrics for the ' + set_names.pop() + ' set:')
                logging.info("Averaged RMSE: {0:{1}f}".format(rmse, prec_rmse))
                logging.info("Averaged MAE: {0:{1}f}\n".format(mae, prec_mae))
                logging.info("Averaged R^2: {0:0.4f}".format(r2))
                outputs.append([rmse, mae, r2])
            else:
                err_pred = np.array(err_pred).ravel()
                # When used for fold/total predictions
                d_r2 = sigma_r2(true, pred, err_pred)
                d_rmse = sigma_rmse(true, pred, err_pred)
                d_mae = sigma_mae(err_pred)
                if len(trues)==1:
                    logging.info("Final cross-validation statistics:")
                else:
                    logging.info("Model performance metrics for the " + set_names.pop() + " set:")

                logging.info("Averaged RMSE: {0:{2}f}+-{1:{2}f}".format(rmse, d_rmse, prec_rmse))
                logging.info("Averaged MAE: {0:{2}f}+-{1:{2}f}".format(mae, d_mae, prec_mae))
                logging.info("Averaged R^2: {0:0.4f}+-{1:0.4f}".format(r2, d_r2))
                logging.info("")

                outputs.append([rmse, d_rmse, mae, d_mae, r2, d_r2])
        elif model_type == 'classification':
            
            print( '\n\n\n\n\n\n\n???', pred[:100], true[:100])
            accuracy, precision, recall, f1, prp_auc, _ = classification_metrics(true, pred)

            prec_acc = output_prec(accuracy, prec)
            prec_prec = output_prec(precision, prec)
            prec_rec = output_prec(recall, prec)
            prec_f1 = output_prec(f1, prec)
            prec_prp_auc = output_prec(prp_auc, prec)

            if err_pred is None:
                logging.info("Averaged accuracy: {0:{1}f}".format(accuracy, prec_acc))
                logging.info("Averaged precision: {0:{1}f}".format(precision, prec_prec))
                logging.info("Averaged recall: {0:{1}f}".format(recall, prec_rec))
                logging.info("Averaged F1: {0:{1}f}".format(f1, prec_f1))
                logging.info("Averaged AUC: {0:{1}f}".format(prp_auc, prec_prp_auc))
                logging.info("")
                outputs.append([accuracy, precision, recall, f1, prp_auc])
            else:                
                err_pred = np.array(err_pred).ravel()
                d_acc, d_prec, d_rec, d_f1, d_prp_auc = sigma_classification_metrics(true, pred, err_pred)

                logging.info("Averaged accuracy: {0:{2}f}+-{1:{2}f}".format(accuracy, d_acc, prec_acc))
                logging.info("Averaged precision: {0:{2}f}+-{1:{2}f}".format(precision, d_prec, prec_prec))
                logging.info("Averaged recall: {0:{2}f}+-{1:{2}f}".format(recall, d_rec, prec_rec))
                logging.info("Averaged F1: {0:{2}f}+-{1:{2}f}".format(f1, d_f1, prec_f1))
                logging.info("Averaged AUC: {0:{2}f}+-{1:{2}f}".format(prp_auc, d_prp_auc, prec_prp_auc))
                logging.info("")

                outputs.append([accuracy, d_acc, precision, d_prec, recall, d_rec, f1, d_f1, prp_auc, d_prp_auc])

    return outputs

# Setup the output format for the dataset automatically, based on the precision requested by user
def output_prec(val, prec):
    
    # Setup the precision of the displayed error to print it cleanly
    if val == 0:
        precision = '0.' + str(prec - 1) # prevent diverging logval if val == 0.
    else:
        logval = np.log10(np.abs(val))
        if logval > 0:
            if logval < prec - 1:
                precision = '1.' + str(int(prec - 1 - np.floor(logval)))
            else:
                precision = '1.0'
        else:
            precision = '0.' + str(int(np.abs(np.floor(logval)) + prec - 1))
    return precision


# Plot individual plots per run for the internal tests
def plot_fit(trues, preds, errs_true, errs_pred, err_bars: str, save_dir: str, dname: str, dlabel: str, units: str, fold: Optional[int] = None, run: Optional[int] = None, final: bool = False, model_type='regression') -> None:
    set_names = ['Test', 'Validation', 'Train']

    if model_type == 'regression':

        fig = plt.figure(figsize=(6.75, 5), dpi=200)

        ax = fig.add_subplot(111)

        # Setting plot limits
        y_true_min = min([t.min() for t in trues])
        y_true_max = max([t.max() for t in trues])
        y_pred_min = min([p.min() for p in preds])
        y_pred_max = max([p.max() for p in preds])

        # Expanding slightly the canvas around the data points (by 10%)
        axmin = y_true_min-0.1*(y_true_max-y_true_min)
        axmax = y_true_max+0.1*(y_true_max-y_true_min)
        aymin = y_pred_min-0.1*(y_pred_max-y_pred_min)
        aymax = y_pred_max+0.1*(y_pred_max-y_pred_min)

        ax.set_xlim(min(axmin, aymin), max(axmax, aymax))
        ax.set_ylim(min(axmin, aymin), max(axmax, aymax))

        colors = ['#cc1b00', '#db702e', '#519fc4']

        if errs_pred is None:
            errs_pred = [None]*len(preds)

        for true, pred, err_true, err_pred in zip(trues, preds, errs_true, errs_pred):
            # Put the shapes of the errors to the format accepted by matplotlib
            # (N, ) for symmetric errors, (2, N) for asymmetric errors
            if err_bars is not None:
                err_true = error_format(true, err_true, err_bars)

            # Legend printing for train/val/test
            if final:
                # No legend is needed for the final out-of-sample prediction
                set_name = None
            else:
                set_name = set_names.pop()

            ax.errorbar(true.ravel(),
                        pred.ravel(),
                        xerr = err_true,
                        yerr = err_pred,
                        fmt='o',
                        label=set_name,
                        ecolor='#bababa',
                        elinewidth = 0.5,
                        ms=5,
                        mfc=colors.pop(),
                        markeredgewidth = 0,
                        alpha=0.7)

        # Define file name
        if final:
            file_name = '{}/Figures/Pred_vs_True/{}_PredvsTrue_Plot_Final.png'.format(save_dir, dname)
        elif run is None:
            file_name = '{}/Figures/Pred_vs_True/Folds/{}_PredvsTrue_Plot_Fold_{}.png'.format(save_dir, dname, fold)
        else:
            file_name = '{}/Figures/Pred_vs_True/Runs/{}_PredvsTrue_Plot_Fold_{}_Run_{}.png'.format(save_dir, dname, fold, run)

        # Plot X=Y line
        ax.plot([max(plt.xlim()[0], plt.ylim()[0]),
                min(plt.xlim()[1], plt.ylim()[1])],
                [max(plt.xlim()[0], plt.ylim()[0]),
                min(plt.xlim()[1], plt.ylim()[1])],
                ':', color = '#595f69')

        if len(units) != 0:
            units = ' (' + units + ')'
        if len(dlabel) != 0:
            plt.xlabel(r"{}, experimental {}".format(dlabel, units), fontsize = 18)
            plt.ylabel(r"{}, prediction {}".format(dlabel, units), fontsize = 18)    
        else:
            plt.xlabel('Ground truth {}'.format(units), fontsize = 18)
            plt.ylabel('Prediction {}'.format(units), fontsize = 18)
        if not final:
            ax.legend(fontsize=14)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(14)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis="x", direction="inout")
        ax.tick_params(axis="y", direction="inout")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis="x", which="minor", direction="out",
            top=True, labeltop=True, bottom=True, labelbottom=True)
        ax.tick_params(axis="y", which="minor", direction="out",
            right=True, labelright=True, left=True, labelleft=True)

        plt.savefig(file_name, bbox_inches='tight')
        plt.close()

    elif model_type == 'classification':
        print( '\n\n\n>>>>>>>> true vs pred', trues[0], preds[0] , '\n\n\n' )
        
        for true, pred, err_pred in zip(trues, preds, errs_pred):
            true, pred = np.array(true).ravel(), np.array(pred).ravel()

            # Legend printing for train/val/test
            if final:
                # No legend is needed for the final out-of-sample prediction
                set_name = 'final'
            else:
                set_name = set_names.pop()

            fig, ax = plt.subplots(figsize=(8, 8), dpi=200)

            accuracy, precision, recall, f1, prp_auc, conf_mat = classification_metrics(true, pred)
            cm = ax.matshow(conf_mat)
            for i in range(conf_mat.shape[0]):
                for j in range(conf_mat.shape[1]):
                    ax.text(x=j, y=i, s=conf_mat[i, j], va='center', ha='center', size='xx-large')
            plt.title(set_name+' set confusion matrix')
            fig.colorbar(cm, ax=ax)
            plt.ylabel('Observed label', fontsize = 18)
            plt.xlabel('Predicted label', fontsize = 18)

            # Define file name
            if final:
                file_name = '{}/Figures/Pred_vs_True/{}_PredvsTrue_{}_ConfMatrix_Final.png'.format(save_dir, set_name, dname)
            elif run is None:
                file_name = '{}/Figures/Pred_vs_True/Folds/{}_PredvsTrue_{}_ConfMatrix_Fold_{}.png'.format(save_dir, set_name, dname, fold)
            else:
                file_name = '{}/Figures/Pred_vs_True/Runs/{}_PredvsTrue_{}_ConfMatrix_Fold_{}_Run_{}.png'.format(save_dir, set_name, dname, fold, run)

            plt.savefig(file_name, bbox_inches='tight')
            plt.close()

            # Precision-recall curve plot 
            # plot the precision-recall curves
            prp_precision, prp_recall, prp_th = precision_recall_curve(true, pred)
            # calculate f score
            fscore = (2 * prp_precision * prp_recall) / (prp_precision + prp_recall)
            # locate the index of the largest f score
            best_fscore_idx = np.argmax(fscore)
            logging.info("Precision-Recall curve best threshold = {0:0.4f}, F1 = {1:0.4f}\n".format(prp_th[best_fscore_idx], fscore[best_fscore_idx]))
            
            random_perf = len(true[true==1]) / len(true)
            plt.plot([0, 1], [random_perf, random_perf], linestyle='--', label='Random')
            plt.plot(prp_recall, prp_precision, marker='.', label='SMILES-X')
            plt.scatter(prp_recall[best_fscore_idx], prp_precision[best_fscore_idx], marker='o', color='black', label='Best F1 score')
            plt.xlabel('Recall', fontsize = 18)
            plt.ylabel('Precision', fontsize = 18)
            plt.legend()

            # Define file name
            if final:
                file_name = '{}/Figures/Pred_vs_True/{}_PredvsTrue_{}_PrecisionRecall_curve_Final.png'.format(save_dir, set_name, dname)
            elif run is None:
                file_name = '{}/Figures/Pred_vs_True/Folds/{}_PredvsTrue_{}_PrecisionRecall_curve_Fold_{}.png'.format(save_dir, set_name, dname, fold)
            else:
                file_name = '{}/Figures/Pred_vs_True/Runs/{}_PredvsTrue_{}_PrecisionRecall_curve_Fold_{}_Run_{}.png'.format(save_dir, set_name, dname, fold, run)

            plt.savefig(file_name, bbox_inches='tight')
            plt.close()
def error_format(val, err, bars):
    # If any error is given
    if err is not None:
        # If one error value is given, it is treated as standard deviation
        if err.shape[1]==1:
            return err.ravel()
        # If two error values are given, they are treated as [min, max]
        elif err.shape[1]==2:
            # Switch from min/max range to the lengths of error bars
            # to the left/right from the mean or median value
            val = val.reshape(-1,1)
            return np.abs(err-val).T
        # If three error values are given, they are treated as [std, min, max]
        elif err.shape[1]==3:
            if bars == 'minmax':
                # Switch from min/max range to the lengths of error bars
                # to the left/right from the mean or median value
                return np.abs(val-err[:,1:]).T
            elif bars == 'std':
                return err[:,0].ravel()
            else:
                logging.warning("ERROR:")
                logging.warning("Error bars format is not understood.")
                logging.warning("")
                logging.warning("SMILES-X execution is aborted.")
                raise StopExecution
    else:
        return err
def sigma_r2(true, pred, err_pred):
    sstot = np.sum(np.square(true - np.mean(true)))
    sigma_r2 = 2/sstot*np.sqrt(np.square(true-pred).T.dot(np.square(err_pred)))
    return float(sigma_r2)
def sigma_rmse(true, pred, err_pred):
    N = float(len(err_pred))
    ssres = np.sum(np.square(true - pred))
    sigma_rmse = np.sqrt(np.square(true-pred).T.dot(np.square(err_pred))/N/ssres)
    return float(sigma_rmse)
def sigma_mae(err_pred):
    N = float(len(err_pred))
    sigma_mae = np.sqrt(np.sum(np.square(err_pred))) / N
    return float(sigma_mae)
def sigma_classification_metrics(true, pred, err_pred, n_mc=1000):
    N = float(len(err_pred))
    sigma = np.zeros((n_mc, 5))
    for i in range(n_mc):
        pred_mc = pred + np.random.normal(0, err_pred)
        sigma[i,0], sigma[i,1], sigma[i,2], sigma[i,3], sigma[i,4], _ = classification_metrics(true, pred_mc)
    sigma = np.std(sigma, axis=0)
    return sigma.ravel()


def main(data_smiles,
         data_prop,
         data_err = None,
         data_extra = None,
         data_name: str = 'Test',
         data_units: str = '',
         data_label: str  = '',
         smiles_concat: bool = False,
         outdir: str = './outputs',
         geomopt_mode: str ='off',
         bayopt_mode: str = 'off',
         train_mode: str = 'on',
         pretrained_data_name: str = '',
         pretrained_augm: str = False,
         model_type = 'regression', 
         scale_output = True, 
         embed_bounds: Optional[List[int]] = None,
         lstm_bounds: Optional[List[int]] = None,
         tdense_bounds: Optional[List[int]] = None,
         nonlin_bounds: Optional[List[int]] = None,
         bs_bounds: Optional[List[int]] = None,
         lr_bounds: Optional[List[float]] = None,
         embed_ref: Optional[int] = 512,
         lstm_ref: Optional[int] = 128,
         tdense_ref: Optional[int] = 128,
         dense_depth: Optional[int] = 0,
         bs_ref: int = 16,
         lr_ref: float = 1e-3,
         k_fold_number: Optional[int] = 5,
         k_fold_index: Optional[List[int]] = None,
         run_index: Optional[List[int]] = None,
         n_runs: Optional[int] = None,
         check_smiles: bool = True,
         augmentation: bool = False,
         geom_sample_size: int = 32,
         bayopt_n_rounds: int = 25,
         bayopt_n_epochs: int = 30,
         bayopt_n_runs: int = 3,
         n_gpus: int = 1,
         gpus_list: Optional[List[int]] = None,
         gpus_debug: bool = False,
         patience: int = 25,
         n_epochs: int = 100,
         batchsize_pergpu: Optional[int] = None,
         lr_schedule: Optional[str] = None,
         bs_increase: bool = False,
         ignore_first_epochs: int = 0,
         lr_min: float = 1e-5,
         lr_max: float = 1e-2,
         prec: int = 4,
         log_verbose: bool = True,
         train_verbose: bool = True) -> None:    
    

    lr_ref = LR0
    start_time = time.time()

    # Define and create output directories
    if train_mode=='finetune':
        save_dir = '{}/{}/{}/Transfer'.format(outdir, data_name, 'Augm' if augmentation else 'Can')
    else:
        save_dir = '{}/{}/{}/Train'.format(outdir, data_name, 'Augm' if augmentation else 'Can')
    scaler_dir = save_dir + '/Other/Scalers'
    model_dir = save_dir + '/Models'
    pred_plot_run_dir = save_dir + '/Figures/Pred_vs_True/Runs'
    pred_plot_fold_dir = save_dir + '/Figures/Pred_vs_True/Folds'
    lcurve_dir = save_dir + '/Figures/Learning_Curves'
    create_dirs = [scaler_dir, model_dir, pred_plot_run_dir, pred_plot_fold_dir, lcurve_dir]
    for create_dir in create_dirs:
        if not os.path.exists(create_dir):
            os.makedirs(create_dir)

    # Setting up logger
    logger, logfile = log_setup(save_dir, "Train", log_verbose)

    logging.info("************************")
    logging.info("***SMILES-X starts...***")
    logging.info("************************")
    logging.info("")
    logging.info("")
    logging.info("The SMILES-X logs can be found in the " + logfile + " file.")
    logging.info("")

    # Reading the data
    header = []
    data_smiles = data_smiles.replace([np.nan, None], ["", ""]).values
    if data_smiles.ndim==1:
        data_smiles = data_smiles.reshape(-1,1)
        header.extend(["SMILES"])
    elif data_smiles.shape[1]==1:
        data_smiles = data_smiles.reshape(-1,1)
        header.extend(["SMILES"])
    else:
        for i in range(data_smiles.shape[1]):
            header.extend(["SMILES_{}".format(i+1)])
    data_prop = data_prop.values
    header.extend([data_label])
    if data_err is not None:
        if data_err.ndim==1:
            data_err = data_err.reshape(-1,1)
        if data_err.shape[1] == 1:
            header.extend(["Standard deviation"])
            err_bars = 'std'
        elif data_err.shape[1] == 2:
            header.extend(["Minimum", "Maximum"])
            err_bars = 'minmax'
        data_err = data_err.values
    else:
        err_bars = None
    if data_extra is not None:
        header.extend(data_extra.columns)
        data_extra = data_extra.values
        extra_dim = data_extra.shape[1]
    else:
        extra_dim = None
    if data_label=='':
        data_label = data_name    

    # Initialize Predictions.txt and Scores.csv files
    predictions = np.concatenate([arr for arr in (data_smiles, data_prop.reshape(-1,1), data_err, data_extra) if arr is not None], axis=1)
    predictions = pd.DataFrame(predictions)
    predictions.columns = header
    scores_folds = []

    logging.info("***Configuration parameters:***")
    logging.info("")
    logging.info("data =\n" + tabulate(predictions.head(), header))
    logging.info("data_name = \'{}\'".format(data_name))
    logging.info("data_units = \'{}\'".format(data_units))
    logging.info("data_label = \'{}\'".format(data_label))
    logging.info("smiles_concat = \'{}\'".format(smiles_concat))
    logging.info("outdir = \'{}\'".format(outdir))
    logging.info("pretrained_data_name = \'{}\'".format(pretrained_data_name))
    logging.info("pretrained_augm = \'{}\'".format(pretrained_augm))
    logging.info("model_type = \'{}\'".format(model_type))
    logging.info("scale_output = \'{}\'".format(scale_output))
    logging.info("geomopt_mode = \'{}\'".format(geomopt_mode))
    logging.info("bayopt_mode = \'{}\'".format(bayopt_mode))
    logging.info("train_mode = \'{}\'".format(bayopt_mode))
    logging.info("embed_bounds = {}".format(embed_bounds))
    logging.info("lstm_bounds = {}".format(lstm_bounds))
    logging.info("tdense_bounds = {}".format(tdense_bounds))
    logging.info("bs_bounds = {}".format(bs_bounds))
    logging.info("lr_bounds = {}".format(lr_bounds))
    logging.info("embed_ref = {}".format(embed_ref))
    logging.info("lstm_ref = {}".format(lstm_ref))
    logging.info("tdense_ref = {}".format(tdense_ref))
    logging.info("dense_depth = {}".format(dense_depth))
    logging.info("bs_ref = {}".format(bs_ref))
    logging.info("lr_ref = {}".format(lr_ref))
    logging.info("k_fold_number = {}".format(k_fold_number))
    logging.info("k_fold_index = {}".format(k_fold_index))
    logging.info("run_index = {}".format(run_index))
    logging.info("n_runs = {}".format(n_runs))
    logging.info("augmentation = {}".format(augmentation))
    logging.info("geom_sample_size = {}".format(geom_sample_size))
    logging.info("bayopt_n_rounds = {}".format(bayopt_n_rounds))
    logging.info("bayopt_n_epochs = {}".format(bayopt_n_epochs))
    logging.info("bayopt_n_runs = {}".format(bayopt_n_runs))
    logging.info("n_gpus = {}".format(n_gpus))
    logging.info("gpus_list = {}".format(gpus_list))
    logging.info("gpus_debug = {}".format(gpus_debug))
    logging.info("patience = {}".format(patience))
    logging.info("n_epochs = {}".format(n_epochs))
    logging.info("batchsize_pergpu = {}".format(batchsize_pergpu))
    logging.info("lr_schedule = {}".format(lr_schedule))
    logging.info("bs_increase = {}".format(bs_increase))
    logging.info("ignore_first_epochs = {}".format(ignore_first_epochs))
    logging.info("lr_min = {}".format(lr_min))
    logging.info("lr_max = {}".format(lr_max))
    logging.info("prec = {}".format(prec))
    logging.info("log_verbose = {}".format(log_verbose))
    logging.info("train_verbose = {}".format(train_verbose))
    logging.info("******")
    logging.info("")

    # Setting up GPUs
    strategy, gpus = set_gpuoptions(n_gpus=n_gpus,
                                          gpus_list=gpus_list,
                                          gpus_debug=gpus_debug)
    if strategy is None:
        raise StopExecution
    logging.info("{} Logical GPU(s) detected and configured.".format(len(gpus)))
    logging.info("")

    # Setting up the scores summary
    scores_summary = {'train': [],
                      'valid': [],
                      'test': []}

    if ignore_first_epochs >= n_epochs:
            logging.error("ERROR:")
            logging.error("The number of ignored epochs `ignore_first_epochs` should be less than")
            logging.error("the total number of training epochs `n_epochs`.")
            logging.error("")
            logging.error("*** SMILES-X EXECUTION ABORTED ***")
            raise StopExecution

    # Retrieve the models for training in case of transfer learning
    if train_mode == 'finetune':
        if len(pretrained_data_name) == 0:
            logging.error("ERROR:")
            logging.error("Cannot determine the pretrained model path.")
            logging.error("Please, specify the name of the data used for the pretraining (`pretrained_data_name`)")
            logging.error("")
            logging.error("*** SMILES-X EXECUTION ABORTED ***")
            raise StopExecution
        if k_fold_number is None:
            # If the dataset is too small to transfer the number of kfolds
            if model.k_fold_number > data.shape[0]:
                k_fold_number = data.shape[0]
                logging.info("The number of cross-validation folds (`k_fold_number`) is not defined.")
                logging.info("Borrowing it from the pretrained model...")
                logging.info("Number of folds `k_fold_number` is set to {}". format(k_fold_number))
            else:
                k_fold_number = model.k_fold_number
                logging.info("The number of cross-validation folds (`k_fold_number`)")
                logging.info("used for the pretrained model is too large to be used with current data:")
                logging.info("size of the data is too small ({} > {})".format(model.k_fold_number, data.shape[0]))
                logging.info("The number of folds is set to the length of the data ({})". format(k_fold_number))
        if n_runs is None:
            logging.info("The number of runs per fold (`n_runs`) is not defined.")
            logging.info("Borrowing it from the pretrained model...")
            logging.info("Number of runs `n_runs` is set to {}". format(model.n_runs))
            
        logging.info("Fine tuning has been requested, loading pretrained model...")
        pretrained_model = loadmodel.LoadModel(data_name = pretrained_data_name,
                                               outdir = outdir,
                                               augmentation = pretrained_augm,
                                               gpu_name = gpus[0].name,
                                               strategy = strategy, 
                                               return_attention=False, # no need to return attention for transfer learning
                                               model_type = model_type,
                                               extra = (data_extra!=None),
                                               scale_output=scale_output, 
                                               k_fold_number = k_fold_number)
    else:
        if k_fold_number is None:
            logging.error("ERROR:")
            logging.error("The number of cross-validation folds (`k_fold_number`) is not defined.")
            logging.error("")
            logging.error("*** SMILES-X EXECUTION ABORTED ***")
            raise StopExecution
        if n_runs is None:
            logging.error("ERROR:")
            logging.error("The number of runs per fold (`n_runs`) is not defined.")
            logging.error("")
            logging.error("*** SMILES-X EXECUTION ABORTED ***")
            raise StopExecution
        pretrained_model = None

    if model_type == 'regression':
        groups = pd.DataFrame(data_smiles).groupby(by=0).ngroup().values.tolist()
        kf = GroupKFold(n_splits=k_fold_number)
        kf.get_n_splits(X=data_smiles, groups=groups)
        kf_splits = kf.split(X=data_smiles, groups=groups)
        model_loss = 'mse'
        model_metrics = [metrics.mae, metrics.mse]
    elif model_type == 'classification':
        scale_output = False
        kf = StratifiedKFold(n_splits=k_fold_number, shuffle=True, random_state=42)
        kf.get_n_splits(X=data_smiles, y=data_prop)
        kf_splits = kf.split(X=data_smiles, y=data_prop)
        model_loss = 'binary_crossentropy'
        model_metrics = ['accuracy']
     
    # Individual counter for the folds of interest in case of k_fold_index
    nfold = 0
    for ifold, (train_val_idx, test_idx) in enumerate(kf_splits):
        start_fold = time.time()

        # In case only some of the folds are requested for training
        if k_fold_index is not None:
            k_fold_number = len(k_fold_index)
            if ifold not in k_fold_index:
                continue
        
        # Keep track of the fold number for every data point
        predictions.loc[test_idx, 'Fold'] = ifold

        # Estimate remaining training duration based on the first fold duration
        if nfold > 0:
            if nfold == 1:
                onefold_time = time.time() - start_time # First fold's duration
            elif nfold < (k_fold_number - 1):
                logging.info("Remaining time: {:.2f} h. Processing fold #{} of data..."\
                             .format((k_fold_number - nfold) * onefold_time/3600., ifold))
            elif nfold == (k_fold_number - 1):
                logging.info("Remaining time: {:.2f} h. Processing the last fold of data..."\
                             .format(onefold_time/3600.))
        logging.info("")
        logging.info("***Fold #{} initiated...***".format(ifold))
        logging.info("")
        
        logging.info("***Splitting and standardization of the dataset.***")
        logging.info("")
        x_train, x_valid, x_test, \
        extra_train, extra_valid, extra_test, \
        y_train, y_valid, y_test, \
        y_err_train, y_err_valid, y_err_test = rand_split(smiles_input = data_smiles,
                                                                prop_input = data_prop,
                                                                extra_input = data_extra,
                                                                err_input = data_err,
                                                                train_val_idx = train_val_idx,
                                                                test_idx = test_idx)
        # Scale the outputs
        if scale_output:
            scaler_out_file = '{}/{}_Scaler_Outputs'.format(scaler_dir, data_name)
            y_train_scaled, y_valid_scaled, y_test_scaled, scaler = robust_scaler(train=y_train,
                                                                                        valid=y_valid,
                                                                                        test=y_test,
                                                                                        file_name=scaler_out_file,
                                                                                        ifold=ifold)
        else:
            y_train_scaled, y_valid_scaled, y_test_scaled, scaler = y_train, y_valid, y_test, None            
            
        # Scale the auxiliary numeric inputs (if given) 
        if data_extra is not None:
            scaler_extra_file = '{}/{}_Scaler_Extra'.format(scaler_dir, data_name)
            extra_train, extra_valid, extra_test, scaler_extra = robust_scaler(train=extra_train,
                                                                                     valid=extra_valid,
                                                                                     test=extra_test,
                                                                                     file_name=scaler_extra_file,
                                                                                     ifold=ifold)
        if 1:
            # Check/augment the data if requested
            train_augm = augm.augmentation(x_train,
                                           train_val_idx,
                                           extra_train,
                                           y_train_scaled,
                                           check_smiles,
                                           augmentation)
            valid_augm = augm.augmentation(x_valid,
                                           train_val_idx,
                                           extra_valid,
                                           y_valid_scaled,
                                           check_smiles,
                                           augmentation)
            test_augm = augm.augmentation(x_test,
                                          test_idx,
                                          extra_test,
                                          y_test_scaled,
                                          check_smiles,
                                          augmentation)        
            x_train_enum, extra_train_enum, y_train_enum, y_train_clean, x_train_enum_card, _ = train_augm
            x_valid_enum, extra_valid_enum, y_valid_enum, y_valid_clean, x_valid_enum_card, _ = valid_augm
            x_test_enum, extra_test_enum, y_test_enum, y_test_clean, x_test_enum_card, test_idx_clean = test_augm            
            
            if DEBUG>1:
                print( '\nTrain enum',  x_train_enum[0], '\nextra_train_enum:', extra_train_enum[0], '\nextra_train_enum:', x_train_enum_card[0], '\ny:', y_train_enum[0], '\nycleaned:',y_train_clean[0])
            ''' 
            >>>> ['C#CC[C@@H](CC(=O)N[Dy])Nc1nc(Nc2ccc(=O)n(C)c2)nc(Nc2ccc(C(C)=O)cc2F)n1'] 0
                extra_train_enum? [[1. 0. 0.]
                 [1. 0. 0.]
                 [1. 0. 0.]
                 ...
                 [0. 1. 0.]
                 [0. 1. 0.]
                 [0. 1. 0.]]
            ''';
        else:            
            if 0:
                x_train_enum_card = smiles_enum_card.extend([1,1] * len(x_train))
                x_train_enum_card = smiles_enum_card.extend([1,1] * len(x_valid))
                x_train_enum_card = smiles_enum_card.extend([1,1] * len(x_test))                       
            extra_train_enum, extra_valid_enum, extra_test_enum = extra_train, extra_valid, extra_test
            x_train_enum, x_valid_enum,x_test_enum=x_train, x_valid, x_test
            y_train_enum,y_valid_enum,y_test_enum=y_train_scaled,y_valid_scaled,y_test_scaled
            
        if DEBUG>2:
            print( 'extra_train_enum?', extra_train_enum)
            
        # Concatenate multiple SMILES into one via 'j' joint
        if smiles_concat:
            x_train_enum = smiles_concat(x_train_enum)
            x_valid_enum = smiles_concat(x_valid_enum)
            x_test_enum =  smiles_concat(x_test_enum)
        
        logging.info("Enumerated SMILES:")
        logging.info("\tTraining set: {}".format(len(x_train_enum)))
        logging.info("\tValidation set: {}".format(len(x_valid_enum)))
        logging.info("\tTest set: {}".format(len(x_test_enum)))
        logging.info("")
        logging.info("***Tokenization of SMILES.***")
        logging.info("")

        # Tokenize SMILES per dataset
        x_train_enum_tokens = token.get_tokens(x_train_enum)
        x_valid_enum_tokens = token.get_tokens(x_valid_enum)
        x_test_enum_tokens = token.get_tokens(x_test_enum)

        logging.info("Examples of tokenized SMILES from a training set:")
        logging.info("{}".format(x_train_enum_tokens[:5]))
        logging.info("")

        # Vocabulary size computation
        all_smiles_tokens = x_train_enum_tokens+x_valid_enum_tokens+x_test_enum_tokens

        # Check if the vocabulary for current dataset exists already
        vocab_file = '{}/Other/{}_Vocabulary.txt'.format(save_dir, data_name)
        if os.path.exists(vocab_file):
            tokens = token.get_vocab(vocab_file)
        else:
            tokens = token.extract_vocab(all_smiles_tokens)
            token.save_vocab(tokens, vocab_file)
            tokens = token.get_vocab(vocab_file)

        train_unique_tokens = token.extract_vocab(x_train_enum_tokens)
        logging.info("Number of tokens only present in training set: {}".format(len(train_unique_tokens)))
        logging.info("")

        valid_unique_tokens = token.extract_vocab(x_valid_enum_tokens)
        logging.info("Number of tokens only present in validation set: {}".format(len(valid_unique_tokens)))
        if valid_unique_tokens.issubset(train_unique_tokens):
            logging.info("Validation set contains no new tokens comparing to training set tokens")
        else:
            logging.info("Validation set contains the following new tokens comparing to training set tokens:")
            logging.info(valid_unique_tokens.difference(train_unique_tokens))
            logging.info("")

        test_unique_tokens = token.extract_vocab(x_test_enum_tokens)
        logging.info("Number of tokens only present in a test set: {}".format(len(test_unique_tokens)))
        if test_unique_tokens.issubset(train_unique_tokens):
            logging.info("Test set contains no new tokens comparing to the training set tokens")
        else:
            logging.info("Test set contains the following new tokens comparing to the training set tokens:")
            logging.info(test_unique_tokens.difference(train_unique_tokens))

        if test_unique_tokens.issubset(valid_unique_tokens):
            logging.info("Test set contains no new tokens comparing to the validation set tokens")
        else:
            logging.info("Test set contains the following new tokens comparing to the validation set tokens:")
            logging.info(test_unique_tokens.difference(valid_unique_tokens))
            logging.info("")

        # Add 'pad' (padding), 'unk' (unknown) tokens to the existing list
        tokens.insert(0,'unk')
        tokens.insert(0,'pad')

        logging.info("Full vocabulary: {}".format(tokens))
        logging.info("Vocabulary size: {}".format(len(tokens)))
        logging.info("")

        # Maximum of length of SMILES to process
        max_length = np.max([len(ismiles) for ismiles in all_smiles_tokens])
        logging.info("Maximum length of tokenized SMILES: {} tokens (termination spaces included)".format(max_length))
        logging.info("")

        # predict and compare for the training, validation and test sets
        x_train_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list=x_train_enum_tokens,
                                                            max_length=max_length + 1,
                                                            vocab=tokens)
        x_valid_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list=x_valid_enum_tokens,
                                                            max_length=max_length + 1,
                                                            vocab=tokens)
        x_test_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list=x_test_enum_tokens,
                                                           max_length=max_length + 1,
                                                           vocab=tokens)
        # Hyperparameters optimisation
        if nfold==0:
            logging.info("*** HYPERPARAMETERS OPTIMISATION ***")
            logging.info("")

            # Dictionary to store optimized hyperparameters
            # Initialize at reference values, update gradually
            hyper_opt = {'Embedding': embed_ref,
                         'LSTM': lstm_ref,
                         'TD dense': tdense_ref,
                         'Batch size': bs_ref,
                         'Learning rate': lr_ref}
            hyper_bounds = {'Embedding': embed_bounds,
                            'LSTM': lstm_bounds,
                            'TD dense': tdense_bounds,
                            'Batch size': bs_bounds,
                            'Learning rate': lr_bounds}
            
            # Geometry optimisation
            if geomopt_mode == 'on':
                geom_file = '{}/Other/{}_GeomScores.csv'.format(save_dir, data_name)
                # Do not optimize the architecture in case of transfer learning
                if train_mode=='finetune':
                    logging.info("Transfer learning is requested together with geometry optimisation,")
                    logging.info("but the architecture is already fixed in the original model.")
                    logging.info("Only batch size and learning rate can be tuned.")
                    logging.info("Skipping geometry optimisation...")
                    logging.info("")
                else:
                    hyper_opt, hyper_bounds = \
                    geomopt.geom_search(data_token=x_train_enum_tokens_tointvec,
                                        data_extra=extra_train_enum,
                                        subsample_size=geom_sample_size,
                                        hyper_bounds=hyper_bounds,
                                        hyper_opt=hyper_opt,
                                        dense_depth=dense_depth,
                                        vocab_size=len(tokens),
                                        max_length=max_length,
                                        geom_file=geom_file,
                                        strategy=strategy, 
                                        model_type=model_type)
            else:
                logging.info("Trainless geometry optimisation is not requested.")
                logging.info("")

             # Bayesian optimisation
            if bayopt_mode == 'on':
                if geomopt_mode == 'on':
                    logging.info("*Note: Geometry-related hyperparameters will not be updated during the Bayesian optimisation.")
                    logging.info("")
                    if not any([bs_bounds, lr_bounds]):
                        logging.info("Batch size bounds and learning rate bounds are not defined.")
                        logging.info("Bayesian optimisation has no parameters to optimize.")
                        logging.info("Skipping...")
                        logging.info("")
                hyper_opt = bayopt.bayopt_run(smiles=data_smiles,
                                              prop=data_prop,
                                              extra=data_extra,
                                              train_val_idx=train_val_idx,
                                              smiles_concat=smiles_concat,
                                              tokens=tokens,
                                              max_length=max_length,
                                              check_smiles=check_smiles,
                                              augmentation=augmentation,
                                              hyper_bounds=hyper_bounds,
                                              hyper_opt=hyper_opt,
                                              dense_depth=dense_depth,
                                              bo_rounds=bayopt_n_rounds,
                                              bo_epochs=bayopt_n_epochs,
                                              bo_runs=bayopt_n_runs,
                                              strategy=strategy,
                                              model_type=model_type, 
                                              scale_output=scale_output, 
                                              pretrained_model=pretrained_model)
            else:
                logging.info("Bayesian optimisation is not requested.")
                logging.info("")
                if geomopt == 'off':
                    logging.info("Using reference values for training.")
                    logging.info("")

            hyper_df = pd.DataFrame([hyper_opt.values()], columns = hyper_opt.keys())
            hyper_file = "{}/Other/{}_Hyperparameters.csv".format(save_dir, data_name)
            hyper_df.to_csv(hyper_file, index=False)

            if (bayopt_mode == 'on') | (geomopt_mode == 'on'):
                logging.info("*** HYPERPARAMETERS OPTIMISATION COMPLETED ***")
                logging.info("")
            
            logging.info("The following hyperparameters will be used for training:")
            for key in hyper_opt.keys():
                if key == "Learning rate":
                    logging.info("    - {}: 10^-{}".format(key, hyper_opt[key]))
                else:
                    logging.info("    - {}: {}".format(key, hyper_opt[key]))
            logging.info("")
            logging.info("File containing the list of used hyperparameters:")
            logging.info("    {}".format(hyper_file))
            logging.info("")

            logging.info("*** TRAINING ***")
            logging.info("")
        start_train = time.time()
        prediction_train_bag = np.zeros((y_train_enum.shape[0], n_runs))
        prediction_valid_bag = np.zeros((y_valid_enum.shape[0], n_runs))
        prediction_test_bag = np.zeros((y_test_enum.shape[0], n_runs))
        
        for run in range(n_runs):
            start_run = time.time()

            # In case only some of the runs are requested for training
            if run_index is not None:
                if run not in run_index:
                    continue

            logging.info("*** Run #{} ***".format(run))
            logging.info(time.strftime("%m/%d/%Y %H:%M:%S", time.localtime()))

            # Checkpoint, Early stopping and callbacks definition
            filepath = '{}/{}_Model_Fold_{}_Run_{}.hdf5'.format(model_dir, data_name, ifold, run)
                
            if train_mode == 'off' or os.path.exists(filepath):
                logging.info("Training was set to `off`.")
                logging.info("Evaluating performance based on the previously trained models...")
                logging.info("")
            else:
                # Create and compile the model
                K.clear_session()
                # Freeze the first half of the network in case of transfer learning
                if train_mode == 'finetune':
                    model_train = model.model_dic['Fold_{}'.format(ifold)][run]
                    # Freeze encoding layers
                    #TODO(Guillaume): Check if this is the best way to freeze the layers as layers' name may differ
                    for layer in mod.layers:
                        if layer.name in ['embedding', 'bidirectional', 'time_distributed']:
                            layer.trainable = False
                    if (nfold==0 and run==0):
                        logging.info("Retrieved model summary:")
                        model_train.summary(print_fn=logging.info)
                        logging.info("")
                elif (train_mode == 'train' or train_mode == 'on'):
                    with strategy.scope():
                        model_train = model.LSTMAttModel.create(input_tokens=max_length+1,
                                                                extra_dim=extra_dim,
                                                                vocab_size=len(tokens),
                                                                embed_units=hyper_opt["Embedding"],
                                                                lstm_units=hyper_opt["LSTM"],
                                                                tdense_units=hyper_opt["TD dense"],
                                                                dense_depth=dense_depth,
                                                                model_type=model_type)
                        custom_adam = Adam(learning_rate=math.pow(10,-float(hyper_opt["Learning rate"])))
                        # model_train.compile(loss=model_loss, optimizer=custom_adam, metrics=model_metrics)

                        model_train.compile(loss=model_loss, optimizer='sgd', metrics= ['acc','FBetaScore'] )
                    
                    if (nfold==0 and run==0):
                        logging.info("Model summary:")
                        model_train.summary(print_fn=logging.info)
                        logging.info("\n")

                batch_size = hyper_opt["Batch size"]
                
                if bs_increase:
                    print( 'bs_increase?', bs_increase)
                    # Apply batch increments schedule in accordance with the paper by S.Smith, Q.Le,
                    # "Don't decay the learning rate, increase the batch size"
                    # https://arxiv.org/abs/1711.00489
                    logging.info("Batch size increment option is selected,")
                    logging.info("learning rate schedule will NOT be applied.")
                    logging.info("")
                    if ignore_first_epochs >= n_epochs:
                        logging.info("The number of epochs to be ignored, `ignore_first_epochs`,")
                        logging.info("should be strictly less than the total number of epochs to train, `n_epochs`.")
                        raise StopExecution

                    # Setting up the batch size schedule
                    # Increment batch twofold every 1/3 of the total of epochs (heuristic)
                    batch_size_schedule = [int(batch_size), int(batch_size*2), int(batch_size*4)]
                    n_epochs_schedule = [int(n_epochs/3), int(n_epochs/3), n_epochs - 2*int(n_epochs/3)]

                    # Fit the model applying the batch size schedule:
                    n_epochs_done = 0
                    best_loss = np.Inf
                    # Keeping track of history
                    # During BS increments model is trained 3 times, histories should be stitched manually
                    history_train_loss = []
                    history_val_loss = []

                    # Define callbacks
                    n_epochs_done = 0
                    best_loss = np.Inf
                    best_epoch = 0
                    logging.info("Training:")
                    for i, batch_size in enumerate(batch_size_schedule):
                        if i == (len(batch_size_schedule) - 1):
                            last = True
                        else:
                            last = False
                        n_epochs_part = n_epochs_schedule[i]
                        # Ignores noise fluctuations of the beginning of the training
                        # Avoids picking up undertrained model
                        # TODO: add early stopping to ignorebeginning
                        ignorebeginning = trainutils.IgnoreBeginningSaveBest(filepath=filepath,
                                                                             n_epochs=n_epochs_part,
                                                                             best_loss=best_loss,
                                                                             best_epoch=best_epoch,
                                                                             initial_epoch=n_epochs_done,
                                                                             ignore_first_epochs=ignore_first_epochs,
                                                                             last=last)
                        logcallback = trainutils.LoggingCallback(print_fcn=logging.info,verbose=train_verbose)
                        
                                                
                        # Default callback list
                        callbacks_list = [ignorebeginning, logcallback]
                        callbacks_list.append(
                            tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            min_delta=0,
                            patience=0,
                            verbose=0,
                            mode='auto',
                            baseline=None,
                            restore_best_weights=False,
                            start_from_epoch=0
                            )
                        )
                        with strategy.scope():
                            if i == 0:
                                logging.info("The batch size is initialized at {}".format(batch_size))
                                logging.info("")
                            else:
                                logging.info("")
                                logging.info("The batch size is changed to {}".format(batch_size))
                                logging.info("")
                            history = model_train.fit(\
                                      trainutils.DataSequence(x_train_enum_tokens_tointvec,
                                                              extra_train_enum,
                                                              props=y_train_enum,
                                                              batch_size=batch_size * strategy.num_replicas_in_sync),
                                      validation_data = \
                                      trainutils.DataSequence(x_valid_enum_tokens_tointvec,
                                                              extra_valid_enum,
                                                              props=y_valid_enum,
                                                              batch_size=batch_size * strategy.num_replicas_in_sync),
                                      shuffle=True,
                                      initial_epoch=n_epochs_done,
                                      epochs=n_epochs_done + n_epochs_part,
                                      callbacks=callbacks_list,
                                      verbose=train_verbose,
                                      max_queue_size=batch_size,
                                      )
                        history_train_loss += history.history['loss']
                        history_val_loss += history.history['val_loss']
                        best_loss = ignorebeginning.best_loss
                        best_epoch = ignorebeginning.best_epoch
                        n_epochs_done += n_epochs_part
                else:
                    
                    print( 'bs_increase==0',)
                    ignorebeginning = trainutils.IgnoreBeginningSaveBest(filepath=filepath,
                                                                         n_epochs=n_epochs,
                                                                         best_loss=np.Inf,
                                                                         initial_epoch=0,
                                                                         ignore_first_epochs=ignore_first_epochs)
                    logcallback = trainutils.LoggingCallback(print_fcn=logging.info,verbose=train_verbose)
                    # Default callback list
                    callbacks_list = [ignorebeginning, logcallback]
                    # Additional callbacks
                    if lr_schedule == 'decay':
                        schedule = trainutils.StepDecay(initAlpha=lr_max,
                                                        finalAlpha=lr_min,
                                                        gamma=0.95,
                                                        epochs=n_epochs)
                        callbacks_list.append(LearningRateScheduler(schedule))
                    elif lr_schedule == 'clr':
                        clr = trainutils.CyclicLR(base_lr=lr_min,
                                                  max_lr=lr_max,
                                                  step_size=8*(x_train_enum_tokens_tointvec.shape[0] // \
                                                              (batch_size//strategy.num_replicas_in_sync)),
                                                  mode='triangular')
                        callbacks_list.append(clr)
                    elif lr_schedule == 'cosine':
                        cosine_anneal = trainutils.CosineAnneal(initial_learning_rate=lr_max,
                                                                final_learning_rate=lr_min,
                                                                epochs=n_epochs)
                        callbacks_list.append(cosine_anneal)

                    # Fit the model
                    with strategy.scope():
                        history = model_train.fit(\
                                      trainutils.DataSequence(x_train_enum_tokens_tointvec,
                                                              extra_train_enum,
                                                              props=y_train_enum,
                                                              batch_size=batch_size * strategy.num_replicas_in_sync),
                                      validation_data = \
                                      trainutils.DataSequence(x_valid_enum_tokens_tointvec,
                                                              extra_valid_enum,
                                                              props=y_valid_enum,
                                                              batch_size=batch_size * strategy.num_replicas_in_sync),
                                      shuffle=True,
                                      initial_epoch=0,
                                      epochs=n_epochs,
                                      callbacks=callbacks_list,
                                      verbose=train_verbose,
                                      batch_size=batch_size,                                      
                                      )
                    history_train_loss = history.history['loss']
                    history_val_loss = history.history['val_loss']

                # Summarize history for losses per epoch
                visutils.learning_curve(history_train_loss, history_val_loss, lcurve_dir, data_name, ifold, run)

                logging.info("Evaluating performance of the trained model...")
                logging.info("")

            print( gpus[0].name, '<< gpu')
            # /job:localhost/replica:0/task:0/device:GPU:0, /job:localhost/replica:0/task:0/device:CPU:0
            #with tf.device(gpus[0].name):
            with tf.device('/device:CPU:0'): # device:GPU:0 

                K.clear_session()
                model_train = load_model(filepath, custom_objects={'SoftAttention': model.SoftAttention()})
                if data_extra is not None:
                    y_pred_train = model_train.predict({"smiles": x_train_enum_tokens_tointvec, "extra": extra_train_enum})
                    y_pred_valid = model_train.predict({"smiles": x_valid_enum_tokens_tointvec, "extra": extra_valid_enum})
                    y_pred_test = model_train.predict({"smiles": x_test_enum_tokens_tointvec, "extra": extra_test_enum})
                else:
                    y_pred_train = model_train.predict({"smiles": x_train_enum_tokens_tointvec})
                    y_pred_valid = model_train.predict({"smiles": x_valid_enum_tokens_tointvec})
                    y_pred_test = model_train.predict({"smiles": x_test_enum_tokens_tointvec})

            # Unscale prediction outcomes
            if scale_output:
                y_pred_train_unscaled = scaler.inverse_transform(y_pred_train.reshape(-1,1)).ravel()
                y_pred_valid_unscaled = scaler.inverse_transform(y_pred_valid.reshape(-1,1)).ravel()
                y_pred_test_unscaled = scaler.inverse_transform(y_pred_test.reshape(-1,1)).ravel()
                
                y_train_clean_unscaled = scaler.inverse_transform(y_train_clean.reshape(-1,1)).ravel()
                y_valid_clean_unscaled = scaler.inverse_transform(y_valid_clean.reshape(-1,1)).ravel()
                y_test_clean_unscaled = scaler.inverse_transform(y_test_clean.reshape(-1,1)).ravel()
            else:
                y_train_clean_unscaled = y_pred_train_unscaled = y_pred_train.ravel()
                y_valid_clean_unscaled = y_pred_valid_unscaled = y_pred_valid.ravel()
                y_test_clean_unscaled = y_pred_test_unscaled = y_pred_test.ravel()

            prediction_train_bag[:, run] = y_pred_train_unscaled
            prediction_valid_bag[:, run] = y_pred_valid_unscaled
            prediction_test_bag[:, run]  = y_pred_test_unscaled            
            
            def mean_result(smiles_enum_card, preds_enum):
                preds_ind = pd.DataFrame(preds_enum, index = smiles_enum_card)
                preds_mean = preds_ind.groupby(preds_ind.index).apply(lambda x: np.mean(x.values)).values.flatten()
                preds_std = preds_ind.groupby(preds_ind.index).apply(lambda x: np.std(x.values)).values.flatten()
                return preds_mean, preds_std                                   
            try:
                # Compute average per set of augmented SMILES for the plots per run
                y_pred_train_mean_augm, y_pred_train_std_augm = mean_result(x_train_enum_card, y_pred_train_unscaled)
                y_pred_valid_mean_augm, y_pred_valid_std_augm = mean_result(x_valid_enum_card, y_pred_valid_unscaled)
                y_pred_test_mean_augm, y_pred_test_std_augm = mean_result(x_test_enum_card, y_pred_test_unscaled)

                
                # Print the stats for the run
                if 0:
                    visutils.print_stats(trues=[y_train_clean_unscaled, y_valid_clean_unscaled, y_test_clean_unscaled],
                                     preds=[y_pred_train_mean_augm, y_pred_valid_mean_augm, y_pred_test_mean_augm],
                                     errs_pred=[y_pred_train_std_augm, y_pred_valid_std_augm, y_pred_test_std_augm],
                                     prec=prec, 
                                     model_type=model_type)

                    # Plot prediction vs observation plots per run
                    visutils.plot_fit(trues=[y_train_clean_unscaled, y_valid_clean_unscaled, y_test_clean_unscaled],
                                  preds=[y_pred_train_mean_augm, y_pred_valid_mean_augm, y_pred_test_mean_augm],
                                  errs_true=[y_err_train, y_err_valid, y_err_test],
                                  errs_pred=[y_pred_train_std_augm, y_pred_valid_std_augm, y_pred_test_std_augm],
                                  err_bars=err_bars,
                                  save_dir=save_dir,
                                  dname=data_name,
                                  dlabel=data_label,
                                  units=data_units,
                                  fold=ifold,
                                  run=run, 
                                  model_type=model_type)
            except Exception as e:
                print(e)

            end_run = time.time()
            elapsed_run = end_run - start_run
            logging.info("Fold {}, run {} duration: {}".format(ifold, run, str(datetime.timedelta(seconds=elapsed_run))))
            logging.info("")            
        # Averaging predictions over augmentations and runs
        pred_train_mean, pred_train_sigma = mean_result(x_train_enum_card, prediction_train_bag)
        pred_valid_mean, pred_valid_sigma = mean_result(x_valid_enum_card, prediction_valid_bag)
        pred_test_mean, pred_test_sigma =  mean_result(x_test_enum_card, prediction_test_bag)

        # Save the predictions to the final table
        predictions.loc[test_idx_clean, 'Mean'] = pred_test_mean.ravel()
        predictions.loc[test_idx_clean, 'Standard deviation'] = pred_test_sigma.ravel()
        predictions.to_csv('{}/{}_Predictions.csv'.format(save_dir, data_name), index=False)
        
        logging.info("Fold {}, overall performance:".format(ifold))

        
        
        
        
        
        
        
        
        
        
        
        
        if 1:
            fold_scores = print_stats(trues=[y_train_clean_unscaled, y_valid_clean_unscaled, y_test_clean_unscaled],
                                               preds=[pred_train_mean, pred_valid_mean, pred_test_mean],
                                               errs_pred=[pred_train_sigma, pred_valid_sigma, pred_test_sigma],
                                               prec=prec, 
                                               model_type=model_type)
            scores_folds.append([err for set_name in fold_scores for err in set_name])

            # Plot prediction vs observation plots for the fold
            if 0:
                plot_fit(trues=[y_train_clean_unscaled, y_valid_clean_unscaled, y_test_clean_unscaled],
                              preds=[pred_train_mean, pred_valid_mean, pred_test_mean],
                              errs_true=[y_err_train, y_err_valid, y_err_test],
                              errs_pred=[pred_train_sigma, pred_valid_sigma, pred_test_sigma],
                              err_bars=err_bars,
                              save_dir=save_dir,
                              dname=data_name,
                              dlabel=data_label,
                              units=data_units,
                              fold=ifold,
                              run=None, 
                              model_type=model_type)
        end_fold = time.time()
        elapsed_fold = end_fold - start_fold
        logging.info("Fold {} duration: {}".format(ifold, str(datetime.timedelta(seconds=elapsed_fold))))
        logging.info("")
        if ifold == (k_fold_number-1) and not k_fold_index:
            logging.info("*******************************")
            logging.info("***Predictions score summary***")
            logging.info("*******************************")
            logging.info("")

            logging.info("***Preparing the final out-of-sample prediction.***")
            logging.info("")
            
            data_prop_clean = data_prop[predictions['Mean'].notna()]
            predictions = predictions.dropna()

            
            print( 'predictions[Mean]', predictions['Mean'], )
            print( 'predictions', predictions, )
            print( 'predictions', predictions.shape  )
            
            if 1: # Print the stats for the whole data
                final_scores = print_stats(trues=[data_prop_clean],
                                                preds=[predictions['Mean'].values],
                                                errs_pred=[predictions['Standard deviation'].values],
                                                prec=prec, 
                                                model_type=model_type)
            
            scores_final = [err for set_name in final_scores for err in set_name]
            
            if 1: # Final plot for prediction vs observation
                plot_fit(trues=[data_prop_clean.reshape(-1,1)],
                              preds=[predictions['Mean'].values],
                              errs_true=[data_err],
                              errs_pred=[predictions['Standard deviation'].values],
                              err_bars=err_bars,
                              save_dir=save_dir,
                              dname=data_name,
                              dlabel=data_label,
                              units=data_units,
                              final=True, 
                              model_type=model_type)
            
            if model_type == 'regression':
                scores_list = ['RMSE', 'MAE', 'R2-score']
            elif model_type == 'classification':
                scores_list = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'PrecisionRecall-AUC']

            scores_folds = pd.DataFrame(scores_folds)
            scores_folds.columns = pd.MultiIndex.from_product([['Train', 'Valid', 'Test'],\
                                                               scores_list,\
                                                               ['Mean', 'Sigma']])
            scores_folds.index.name = 'Fold'
            scores_folds.to_csv('{}/{}_Scores_Folds.csv'.format(save_dir, data_name))
            
            scores_final = pd.DataFrame(scores_final).T
            scores_final.columns = pd.MultiIndex.from_product([scores_list,\
                                                               ['Mean', 'Sigma']])
            scores_final.to_csv('{}/{}_Scores_Final.csv'.format(save_dir, data_name), index=False)        
        nfold += 1        
    logging.info("*******************************************")
    logging.info("***SMILES_X has terminated successfully.***")
    logging.info("*******************************************")

    end_all = time.time()
    elapsed_tot = end_all - start_time
    logging.info("Total elapsed time: {}".format(str(datetime.timedelta(seconds=elapsed_tot))))
    logging.shutdown()    
    
# ---------------------------------------- Defining the hyperparameters' bounds ----------------------------------------
embed_bounds = [8, 16] # embedding size
lstm_bounds = [8, 16] # number of units in the LSTM layer
tdense_bounds = [8, 16] # number of units in the dense layer

embed_bounds = [256] # embedding size
lstm_bounds = [512] # number of units in the LSTM layer
tdense_bounds = [128] # number of units in the dense layer


bs_bounds = [128,256] # batch size

lr_bounds = [2., 2.5, 3., 3.5] # learning rate

data_name = 'Test'
data_label = 'Test label' # will show on plots
data_units = 'units' # will show on plots

            

devices = tf.config.list_logical_devices('GPU')
NGPUS=len(devices)
print( '\n\n\n>>>>>>>>>>>>>>NGPUs',NGPUS )




main(data_smiles=X['trn'][ F ],    # SMILES input
          data_extra=X['trn'][ E ],                # Aux input 
          data_prop= Y['trn'],                     # prediction
          bs_ref = BS,
          lr_ref = LR,
          data_label=data_label,
          smiles_concat=False,                     # was true in demo 
          geomopt_mode='off',                      #  'off' <-- was off in demo code: Zero-cost geometry optimization
          bayopt_mode='off',                       # Bayesian optimization
          train_mode='on',                         # Train
          model_type = 'classification',           # 'regression'
          scale_output = True,          
          k_fold_number=NFOLDS,                         # Number of cross-validation splits
          n_runs=NRUNS,                                # Number of runs per fold
          check_smiles=False,                       # Verify SMILES validity via RDKit
          augmentation=AUG,                        # Augment the data or not
          bayopt_n_rounds=2,
          bayopt_n_epochs=2,                       # 5
          bayopt_n_runs=2,                         # 2 
          n_gpus=NGPUS,
          n_epochs=NEPOCHS,                              # 10
          log_verbose=True,                        # To send print outs both to the file and console
          train_verbose=True,                     # Show model training progress
          bs_bounds=bs_bounds,          lr_bounds=lr_bounds,          embed_bounds=embed_bounds,          lstm_bounds=lstm_bounds,          tdense_bounds=tdense_bounds,
          data_name=data_name,          data_units=data_units,     
    )     





class LoadModel:    
    def __init__(self,
                 data_name: str,
                 augment: bool,
                 outdir: str = "./outputs",
                 use_cpu: bool = False,
                 gpu_ind: int = 0,
                 gpu_name: str = None,
                 strategy = None,
                 log_verbose: bool = True,
                 return_attention: bool = True):
        self.data_name = data_name
        self.augment = augment
        self.outdir = outdir
        self.gpu_ind = gpu_ind
        self.log_verbose = log_verbose
        self.return_attention = return_attention

        self.train_dir = "{}/{}/{}/Train".format(self.outdir, self.data_name, 'Augm' if self.augment else 'Can')
        if gpu_name is not None:
            print( f'use {gpu_name}')
            self.gpus = gpu_name
            self.strategy = strategy
        elif use_cpu:
            print( 'use cpu')
            # CPUs options
            self.strategy, self.gpus = set_gpuoptions(n_gpus=0,
                                                            gpus_debug=False,
                                                            print_fn=print)
        else:
            print( 'use gpu, as many times as needed')

            # GPUs options
            self.strategy, self.gpus = set_gpuoptions(n_gpus=1,
                                                            gpus_list=[gpu_ind],
                                                            gpus_debug=False,
                                                            print_fn=print)
        # Verify path existance
        if not os.path.exists(self.train_dir):
            print("ERROR:")
            print("Path {} does not exist.\n".format(self.train_dir))
            print("HINT: check the data name and the augmentation flag.\n")
            print("")
            print("*** LOADING ABORTED ***")
            raise StopExecution
            
        # Verify existance of vocabulary file
        vocab_file = '{}/Other/{}_Vocabulary.txt'.format(self.train_dir, self.data_name)
        if not os.path.exists(vocab_file):
            print("ERROR:")
            print("The input directory does not contain any vocabulary (*_Vocabulary.txt file).\n")
            print("")
            print("*** LOADING ABORTED ***")
            raise StopExecution
        else:
            self.vocab_file = vocab_file
            
        scaler_dir = self.train_dir + '/Other/Scalers'
        model_dir = self.train_dir + '/Models'
        
        # Check whether additional data have been used for training
        self.scale_output = False
        n_output_scalers = len(glob.glob(scaler_dir + "/*Outputs*"))
        if n_output_scalers != 0:
            self.scale_output = True
        self.extra = False
        n_extra_scalers = len(glob.glob(scaler_dir + "/*Extra*"))
        if n_extra_scalers != 0:
            self.extra = True
        
        n_models = len(glob.glob(model_dir + "/*"))
        self.k_fold_number = len(glob.glob(model_dir + "/*Model_Fold_*_Run_0.hdf5"))
        self.n_runs = len(glob.glob(model_dir + "/*Model_Fold_0_Run_*.hdf5"))

        print("\nAll the required model files have been found.")
        
        # Load tokens from vocabulary file
        self.tokens = token.get_vocab(self.vocab_file)
        # Add 'pad', 'unk' tokens to the existing list
        self.tokens.insert(0, 'unk')
        self.tokens.insert(0, 'pad')
        
        # Setting up the scalers, trained models, and vocabulary
        self.att_dic = {}
        self.model_dic = {}
        if self.scale_output:
            self.output_scaler_dic = {}
        if self.extra:
            self.extra_scaler_dic = {}
        
        # Start loading models
        for ifold in range(self.k_fold_number):
            fold_model_list = []
            fold_att_list = []
            for run in range(self.n_runs):
                K.clear_session()
                model_file = '{}/{}_Model_Fold_{}_Run_{}.hdf5'.format(model_dir, self.data_name, ifold, run)
                model_tmp = load_model(model_file, custom_objects={'SoftAttention': model.SoftAttention()})
                fold_model_list.append(model_tmp)                                
                
                config = model_tmp.get_config() 

                
                # Retrieve max_length
                if ifold == 0 and run == 0:
                    
                    self.max_length = config['layers'][0]["config"]["batch_shape"][1]

                # For the attention, collect truncated
                if self.return_attention:
                    # Retrieve the geometry based on the trained model                    
                    '''
                    config['layers'][-5]  # time_distributed 
                    config['layers'][-6] # bidirectional ['build_config']['input_shape']
                    config['layers'][-7] # embedding ['build_config']['input_shape']
                    '''
                    
                    embed_att = config['layers'][-7]['config']['output_dim']   
                    lstm_att = config['layers'][-4] ['build_config']['input_shape'][-1]  //2
                    tdense_att = config['layers'][-5]["config"]['layer']['config']['units']

                    # Architecture to return attention weights
                    K.clear_session()
                    att_tmp = model.LSTMAttModel.create(input_tokens=self.max_length,
                                                        vocab_size=len(self.tokens),
                                                        embed_units=embed_att,
                                                        lstm_units=lstm_att,
                                                        tdense_units=tdense_att,
                                                        dense_depth=0,
                                                        return_prob=True)
                    att_tmp.load_weights(model_file, by_name=True, skip_mismatch=True)                                       
                    
                    att_config = att_tmp.get_config() 
                    #print( '>>>>>>>>\n', att_config )
                    #print( att_tmp.layers, '\n\n\n\n' )
                    print( dir( att_tmp.get_layer('smiles')) )
                    
                    intermediate_layer_model = Model(inputs=att_tmp.get_layer('smiles').input,
                                                     outputs=att_tmp.get_layer('attention').output)
                    fold_att_list.append(intermediate_layer_model)
            
            # Save models for the current fold
            self.model_dic['Fold_{}'.format(ifold)] = fold_model_list
            # Save truncated models for the current fold if requested
            if self.return_attention:
                self.att_dic['Fold_{}'.format(ifold)] = fold_att_list
            
            # Collect the scalers
            if self.scale_output:
                output_scaler_file = '{}/{}_Scaler_Outputs_Fold_{}.pkl'.format(scaler_dir, self.data_name, ifold)
                self.output_scaler_dic["Fold_{}".format(ifold)] = load(open(output_scaler_file, 'rb'))
            if self.extra:
                extra_scaler_file = '{}/{}_Scaler_Extra_Fold_{}.pkl'.format(scaler_dir, self.data_name, ifold)
                self.extra_scaler_dic["Fold_{}".format(ifold)] = load(open(extra_scaler_file, 'rb'))
                
        print("\n*** MODELS LOADED ***")

# Load trained models once, use as many times as needed
fitted_model = LoadModel(data_name=data_name,
                            augment=AUG,
                            gpu_ind=0,
                            return_attention=False )


if DEBUG>0:
    # Read the test.parquet file into a pandas DataFrame
    for df_test in pd.read_csv(test_file, chunksize=10000):

        df_test['protein_code1'] = (df_test['protein_name'] == 'BRD4').astype(np.int8)
        df_test['protein_code2'] = (df_test['protein_name'] == 'HSA' ).astype(np.int8)
        df_test['protein_code3'] = (df_test['protein_name'] == 'sEH' ).astype(np.int8)
        break
def infer(model, data_smiles, data_extra=None, augment=False, check_smiles: bool = True, smiles_concat: bool = False, log_verbose: bool = True):
    save_dir =  "{}/{}/{}/Inference/{}".format(model.outdir,
                                               model.data_name,
                                               'Augm' if model.augment else 'Can',
                                               'Augm' if augment else 'Can')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger, logfile = utils.log_setup(save_dir, 'Inference', log_verbose)
    
    logging.info("*************************************")
    logging.info("***   SMILESX INFERENCE STARTED   ***")
    logging.info("*************************************")
    logging.error("")
    
    logging.info("Inference logs path:")
    logging.info(logfile)
    logging.info("")

    if model.extra and data_extra is None:
        logging.error("ERROR:")
        logging.error("Additional input data has been used during the training of the loaded")
        logging.error("model, but none are provided for inference. Please, use `data_extra`")
        logging.error("to provide additional data.")
        logging.error("")
        logging.error("*** INFERENCE ABORTED ***")
        raise utils.StopExecution
        
    logging.info("Full vocabulary: {}".format(model.tokens))
    logging.info("Vocabulary size: {}".format(len(model.tokens)))
    logging.info("Maximum length of tokenized SMILES: {} tokens.\n".format(model.max_length))

    data_smiles = np.array(data_smiles)
    if model.extra:
        data_extra = np.array(data_extra)
    # Checking and/or augmenting the SMILES if requested

    smiles_enum, extra_enum, _, _, smiles_enum_card, _ = augm.augmentation(data_smiles=data_smiles,
                                                                     indices=[i for i in range(len(data_smiles))],
                                                                     data_extra=data_extra,
                                                                     data_prop=None,
                                                                     check_smiles=check_smiles,
                                                                     augment=augment)

    # Concatenate multiple SMILES into one via 'j' joint
    if smiles_concat:
        smiles_enum = utils.smiles_concat(smiles_enum)
        
    logging.info("Number of enumerated SMILES: {}".format(len(smiles_enum)))
    logging.info("")
    logging.info("Tokenization of SMILES...")
    smiles_enum_tokens = token.get_tokens(smiles_enum)
    smiles_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list=smiles_enum_tokens,
                                                       max_length=model.max_length,
                                                       vocab=model.tokens)
    # Model ensembling
    preds_enum = np.empty((len(smiles_enum), model.k_fold_number*model.n_runs), dtype='float')
    for ifold in range(model.k_fold_number):
        # Scale additional data if provided
        if model.extra:
            # Load the scalers from pickle
            data_extra = model.extra_scaler_dic["Fold_{}".format(ifold)].transform(extra_enum)
        for run in range(model.n_runs):
            imodel = model.model_dic["Fold_{}".format(ifold)][run]
            # Predict and compare for the training, validation and test sets
            # Compute a mean per set of augmented SMILES
            if model.extra:
                ipred = imodel.predict({"smiles": smiles_enum_tokens_tointvec, "extra": extra_enum})
            else:
                ipred = imodel.predict({"smiles": smiles_enum_tokens_tointvec})
            if model.scale_output:
                # Unscale predictions
                ipred_unscaled = model.output_scaler_dic["Fold_{}".format(ifold)].inverse_transform(ipred.reshape(-1,1))
            else:
                ipred_unscaled = ipred
            # Store predictions in an array
            preds_enum[:, ifold * model.n_runs + run] = ipred_unscaled.flatten()

    preds_mean, preds_std = utils.mean_result(smiles_enum_card, preds_enum)

    preds = pd.DataFrame()
    if  data_smiles.shape[1]>1:
        preds['SMILES'] = pd.DataFrame(data_smiles[:,0]) 
    else:
        preds['SMILES'] = pd.DataFrame(data_smiles)
    print( preds.head() )
    print( preds_std, preds_mean )
    preds['mean'] = preds_mean
    preds['sigma'] = preds_std
    logging.info("")
    logging.info("Prediction results:\n" \
                 + tabulate(preds, ['SMILES', 'Prediction (mean)', 'Prediction (std)']))
    logging.info("")

    logging.info("***************************************")
    logging.info("***   SMILESX INFERENCE COMPLETED   ***")
    logging.info("***************************************")

    return preds



# usecols=['id','protein_name']+F,
for i, df_test in enumerate(pd.read_csv(test_file, chunksize=100000 )):
    
    df_test['protein_code1'] = (df_test['protein_name'] == 'BRD4').astype(np.int8)
    df_test['protein_code2'] = (df_test['protein_name'] == 'HSA' ).astype(np.int8)
    df_test['protein_code3'] = (df_test['protein_name'] == 'sEH' ).astype(np.int8)

    preds = infer(  model=fitted_model,
                    data_smiles=df_test[ F ],
                    data_extra=df_test[ E ],
                    augment=AUG,
                    check_smiles=False,
                    log_verbose=True)        

    output_df = pd.DataFrame({'id': df_test['id'], 'binds': preds['mean']  })        
    try:
        all_df = pd.DataFrame({'id': df_test['id'],  'p1': df_test['protein_code1'], 'p2': df_test['protein_code2'], 'p3': df_test['protein_code3'], 'binds': preds['mean'], 'molecule_smiles': df_test['molecule_smiles']   })        
    except:
        pass
        
    # Save the output DataFrame to a CSV file
    output_df.to_csv(output_file, index=False, mode='a', header=not os.path.exists(output_file))

    print( f'Written submission output for batch {i}' )
    print( output_df.head(10) )

try:
    pritn( '\n\nMissing values?',)
    print( all_df[all_df.isnull().any(axis=1)]  )
except:
    pass
