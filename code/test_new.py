
import sys, os; 
sys.path.append( '/kaggle/input/smilesx-demo/SMILES-X/')

# demo data from SMILESX
data_dir = '/kaggle/input/smilesx-demo/SMILES-X/data/'

# data from competition 
train_path = '/kaggle/input/leash-BELKA/train.parquet'
test_path = '//kaggle/input/leash-BELKA/test.parquet'
test_file = '/kaggle/input/leash-BELKA/test.csv'
output_file = 'submission.csv'  # Specify the path and filename for the output file


# !conda env export | grep -v "^prefix: " > environment.yml 

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

logger = logging.getLogger()


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

np.random.seed(seed=123)
np.set_printoptions(precision=3)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # was '3'   # Suppress Tensorflow warnings
tf.autograph.set_verbosity(3)
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)














import sys, os; 

# demo data from SMILESX
data_dir = '/kaggle/input/smilesx-demo/SMILES-X/data/'

# data from competition 
train_path = '/kaggle/input/leash-BELKA/train.parquet'
test_path = '//kaggle/input/leash-BELKA/test.parquet'

# !conda env export | grep -v "^prefix: " > environment.yml 



test_file = '/kaggle/input/leash-BELKA/test.csv'
output_file = 'submission.csv'  # Specify the path and filename for the output file

if DEBUG>0:
    # Read the test.parquet file into a pandas DataFrame
    for df_test in pd.read_csv(test_file, chunksize=10000):

        df_test['protein_code1'] = (df_test['protein_name'] == 'BRD4').astype(np.int8)
        df_test['protein_code2'] = (df_test['protein_name'] == 'HSA' ).astype(np.int8)
        df_test['protein_code3'] = (df_test['protein_name'] == 'sEH' ).astype(np.int8)
        break
        
        

        
        

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

# from SMILESX import utils, model, token, augm


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

logger = logging.getLogger()


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
from tensorflow.keras import backend as K

from sklearn.model_selection import GroupKFold, StratifiedKFold

try:
    from SMILESX import utils, token, augm
    from SMILESX import model, bayopt, geomopt
    from SMILESX import visutils, trainutils
    from SMILESX import loadmodel
except:
    pass


from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder

import duckdb
import pandas as pd


data_name = 'Test'

np.random.seed(seed=123)
np.set_printoptions(precision=3)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # was '3'   # Suppress Tensorflow warnings
tf.autograph.set_verbosity(3)
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)





if TRAIN:    
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
    
    if ('df1' in globals())==False:
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
    for t in ['trn', 'val']:
        Y[t]=Y[t].astype( np.int8 )
    print( Y['trn'].head() )
    
    

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


 
data_name = 'Test'
data_label = 'Test label' # will show on plots
data_units = 'units' # will show on plots

            
import tensorflow as tf
devices = tf.config.list_logical_devices('GPU')
NGPUS=len(devices)
print( '\n\n\n>>>>>>>>>>>>>>NGPUs',NGPUS )

data_label='binded?'

if 0:
    main(data_smiles=X['trn'][ F ],    # SMILES input
          data_extra=X['trn'][ E ],                # Aux input 
          data_prop= Y['trn'],                     # prediction
          
          data_label=data_label,
          smiles_concat=False,                     # was true in demo 
          geomopt_mode='off',                      #  'off' <-- was off in demo code: Zero-cost geometry optimization
          bayopt_mode='off',                       # Bayesian optimization
          train_mode='on',                         # Train
          model_type = 'classification',           # 'regression'
          scale_output = True,          
          k_fold_number=NFOLDS,                         # Number of cross-validation splits
          n_runs=NRUNS,                                # Number of runs per fold
          check_smiles=True,                       # Verify SMILES validity via RDKit
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
                 outdir: str = f"{IMPORT_DIR}/outputs",
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

        import glob
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
        from SMILESX import token

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
        
        
def infer(model, data_smiles, data_extra=None, augment=False, check_smiles: bool = True, smiles_concat: bool = False, log_verbose: bool = True):
    save_dir =  "{}/{}/{}/Inference/{}".format(model.outdir,
                                               model.data_name,
                                               'Augm' if model.augment else 'Can',
                                               'Augm' if augment else 'Can')
    
    LOG=1
    if TRAIN & (not os.path.exists(save_dir)):
        os.makedirs(save_dir)
    if TRAIN==0:
        LOG=0

    if LOG:
        logger, logfile = utils.log_setup(save_dir, 'Inference', log_verbose)

        logging.info("*************************************")
        logging.info("***   SMILESX INFERENCE STARTED   ***")
        logging.info("*************************************")
        logging.error("")

        logging.info("Inference logs path:")
        logging.info(logfile)
        logging.info("")

    if LOG and model.extra and data_extra is None:
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

    if LOG:
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
    
    if LOG:
        logging.info("")
        logging.info("Prediction results:\n" \
                     + tabulate(preds, ['SMILES', 'Prediction (mean)', 'Prediction (std)']))
        logging.info("")

        logging.info("***************************************")
        logging.info("***   SMILESX INFERENCE COMPLETED   ***")
        logging.info("***************************************")

    return preds





for i, df_test in enumerate(pd.read_csv(test_file,usecols=['id','protein_name']+F, chunksize=100000 )):
    if 1:
        df_test['protein_code1'] = (df_test['protein_name'] == 'BRD4').astype(np.int8)
        df_test['protein_code2'] = (df_test['protein_name'] == 'HSA' ).astype(np.int8)
        df_test['protein_code3'] = (df_test['protein_name'] == 'sEH' ).astype(np.int8)

        preds = infer(  model=fitted_model,
                        data_smiles=df_test[ F ],
                        data_extra=df_test[ E ],
                        augment=False,
                        check_smiles=False,
                        log_verbose=False)        

        output_df = pd.DataFrame({'id': df_test['id'], 'binds': preds['mean']  })        
        output_df.to_csv(output_file, index=False, mode='a', header=not os.path.exists(output_file))

        print( f'Written submission output for batch {i}' )
        print( output_df.head(10) )

