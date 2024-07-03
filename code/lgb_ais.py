# source:  https://www.kaggle.com/code/pjbhaumik/0-453-leash-bio-predict-new-medicines-with-belka
import timeit
import os, sys

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder

import duckdb
import pandas as pd
import numpy as np

from pyarrow.parquet import ParquetFile
import pyarrow as pa 

from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin

from tqdm.auto import tqdm

import lightgbm as lgb
import optuna 

from rdkit.Chem import MolFromSmiles, MolToSmiles
import atomInSmiles

import sentencepiece as spm        
import sys, os
import requests 

def center_pad( a, N=500 ):   
    m = len(a)    
    n = np.abs(N - m)//2 
    p = [0]*n+a+[0]*n if m<N else a[n:-n]       
    #if (N-len(p))>0:
    #    p=[0]+p
    return p #np.asarray(p, dtype=np.int32)


if feature == 'ais':
    sys.path.append( '/kaggle/working/atom-in-SMILES/atomInSmiles')
    sys.path.append( '/kaggle/working/atom-in-SMILES/')
    sys.path.append( '/kaggle/working/atom-in-SMILES/utils')
    
    # train sentencepiece model from `botchan.txt` and makes `m.model` and `m.vocab`
    # `m.vocab` is just a reference. not used in the segmentation.
    
    r = requests.get('https://raw.githubusercontent.com/google/sentencepiece/master/data/botchan.txt' )
    if r.status_code == 200:
        with open("botchan.txt", "wb") as file:
            file.write( r.content )            
    spm.SentencePieceTrainer.train('--input=botchan.txt --model_prefix=m --vocab_size=2000')    
    sp_encoder = spm.SentencePieceProcessor()
    sp_encoder.load('m.model')

pad_end = lambda a,i=600: a[0:i] if len(a) > i else a + [0] * (i-len(a))
pad_start = lambda a,i=400: a[0:i] if len(a) > i else [0] * (i-len(a))+a

def extract(df, feature):  # ======================================================================          
    L = [feature]
    print( 'After calling extract:', df.head(3) )
    print( feature )
    if feature == 'ais':
        df[feature] = df['molecule_smiles'].progress_apply(generate_ais).progress_apply( sp_encoder.EncodeAsIds ).progress_apply( pad_start )   
    elif feature == 'ecfp_bbs':
        b1=df['buildingblock1_smiles'].progress_apply(Chem.MolFromSmiles).progress_apply(generate_ecfp)
        b2=df['buildingblock2_smiles'].progress_apply(Chem.MolFromSmiles).progress_apply(generate_ecfp)
        b3=df['buildingblock3_smiles'].progress_apply(Chem.MolFromSmiles).progress_apply(generate_ecfp)                                      
        df[feature] = b1+b2+b3
    elif feature == 'ecfp':           
        if SRC_SMILES:
            df[feature] = df['molecule_smiles'].progress_apply(generate_ecfp)
        else:                
            df['molecule'] = df['molecule_smiles'].progress_apply(Chem.MolFromSmiles)        
            df[feature] = df['molecule'].progress_apply(generate_ecfp)
    try:
        return df[ L + ['molecule_smiles', 'binds']]
    except:
        return df 

def generate_ecfp(molecule, radius=2, bits=1024): # ======================================================================
    if molecule is None:
        return None
    return list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits))

def generate_ais(smiles): # ======================================================================
    return atomInSmiles.encode(smiles, with_atomMap=True)
    
def get_train_set(feature = 'ecfp', N=100, N2=200): # 30000       
    def get_data(target):                       
        feature_list = { 'ecfp_bbs': ['buildingblock1_smiles','buildingblock2_smiles', 'buildingblock3_smiles'], 
            'ecfp': ['molecule_smiles'],'ais': ['molecule_smiles']} 
        if N+N2<10000:
            pf = ParquetFile(train_path) 
            first_ten_rows = next(pf.iter_batches(batch_size = N+N2)) 
            df = pa.Table.from_batches([first_ten_rows]).to_pandas()             
        else:
            if (N+N2)>1000:
                con = duckdb.connect()    
            df = con.query(f"""(SELECT {feature_list[feature]}, binds
                                FROM parquet_scan('{train_path}')
                                WHERE binds = 0
                                and protein_name = '{target}'
                                ORDER BY random()
                                LIMIT {N})
                                UNION ALL
                                (SELECT {feature_list[feature]}, binds
                                FROM parquet_scan('{train_path}')
                                WHERE binds = 1
                                and protein_name = '{target}'
                                ORDER BY random()
                                LIMIT {N2})""").df()               
            if (N+N2)>1000:
                print('connection closed')
                con.close()     
        print( 'Before calling extract:', df.head(3) )
        return extract(df, feature)          
    return list(map(get_data, targets))

class VotingModel(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators        
    def fit(self, X, y=None):
        return self    
    def predict(self, X):
        y_preds = [estimator.predict(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)    
    def predict_proba(self, X):
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)    

def apply_to_unseen(models, df, feature ):    
    df_final = extract( df, feature )             
    print( models.keys() )    
    df_final['binds'] = df_final.progress_apply(lambda x: models[x['protein_name']].predict_proba(np.array(x[feature]).reshape(1, -1))[:,1][0], axis = 1)    
    return df_final[['id', 'binds']]    

def fit_models(df):      
    X = pd.DataFrame(df[feature].to_list())
    y = df['binds']    
    print( X.head(3), '\n', y )
    fitted_models = []
    for idx_train, idx_valid in skf.split(X, y):
        X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
        X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]    
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.log_evaluation( NEPOCHS ), lgb.early_stopping( NEPOCHS )])        
        fitted_models.append(model)
    return VotingModel(fitted_models)    
#if __name__ != "__main__":  # called when imported
if 1:                      
    test_file = os.path.dirname(test_path) + '/test.csv'    
    feat_functions = {'ecfp': generate_ecfp, 'ais':generate_ais }         
    if ('data' in globals())==False:               
        tqdm.pandas()  # enable progress_apply   
        targets = ['BRD4', 'sEH', 'HSA']                
        if DEBUG:            
            N, N2, NE, NEPOCHS = 100,300,10,5            
            print( 'num of ests + adjusted to mini training set')            
        tm_start = timeit.default_timer()        
        
        data = get_train_set( feature, N=N, N2=N2)               
        
        data = dict(zip(targets, data))            
        tm_end = timeit.default_timer()
        rtime = (tm_end - tm_start)/ 60
        print( f'\n\n********* Reading completed in {rtime:.2f} min. *********' )        

if ('TRAIN' in globals())==False:
    TRAIN = True
if TRAIN:   
    # Split the data into train and test sets
    skf = StratifiedKFold(n_splits=3, shuffle=False)
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": 'average_precision',
        "max_depth": 8,
        "learning_rate": 0.05,
        "n_estimators": NE,
        "colsample_bytree": 0.8, 
        "colsample_bynode": 0.8,
        "verbose": -1,
        "random_state": 42,}
        
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')  
    if len(gpus):
        params.update( {"device": "gpu",})
 
    tm_start = timeit.default_timer()
    print( '\n\n********* Training begins...' )
    models = {k: fit_models(v) for k, v in data.items()}
        
    tm_end = timeit.default_timer()        
    rtime = (tm_end - tm_start)/ 60
    print( f'\n\n********* Training completed in {rtime:.2f}*********' )
     
