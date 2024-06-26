#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-06-25 

@author: r4g
"""

from sklearn import metrics
from sklearn.metrics import average_precision_score, fbeta_score, make_scorer

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score

import optuna
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import catboost
from catboost import CatBoostClassifier
from optuna.integration import CatBoostPruningCallback

import xgboost
from xgboost import XGBClassifier
import joblib 


# ===================== define custom scorers =====================

avg_pres_score_pos = make_scorer(average_precision_score,
                             greater_is_better = True,
                             needs_proba = True,
                             pos_label=0)

fbeta_scorer = make_scorer(fbeta_score,
                          beta=1,
                          greater_is_better = True,
                          pos_label=0)

# ===================== myOptuna  =====================
class myOptuna:    
    def __init__(self, name, X_train, y_train, X_val, y_val ):
        self.name = name         
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
            
    def rf_objective(self, trial):
        #n_estimators = trial.suggest_int('n_estimators', 1, 2)
        n_estimators = trial.suggest_int("n_estimators", 100, 300, log=True)

        max_depth = trial.suggest_int('max_depth', 10, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 32)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 32)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        #criterion = trial.suggest_categorical('criterion', [avg_pres_score_pos, fbeta_scorer])    
        criterion = trial.suggest_categorical('criterion', ['entropy', 'log_loss', 'gini'])

        # criterion = trial.suggest_categorical('criterion', ["squared_error", "absolute_error", "friedman_mse", "poisson"])

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
            random_state= 21
        )

        model.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], early_stopping_rounds=100,)    
        y_pred = model.predict(self.X_val)
        pred_labels = np.rint(y_pred)        
        return accuracy_score(self.y_val, pred_labels) 
    


    def cat_objective(self, trial):
        param = {
            "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 1, 12),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
            "eval_metric": "Accuracy",
        }
        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

        model = catboost.CatBoostClassifier(**param)    
        
        model.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], early_stopping_rounds=100,)    
        y_pred = model.predict(self.X_val)
        pred_labels = np.rint(y_pred)
        
        # metric  to optimize
        #  mean_squared_error(y_test, y_pred)        
        return accuracy_score(self.y_val, pred_labels) 
    

    def run(self,n_trials=200):
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=42))
      
        if self.name == 'cat':
            study.optimize(self.cat_objective, n_trials=n_trials)            
        elif self.name == 'rf':
            study.optimize(self.rf_objective, n_trials=n_trials)                         

        # Print the best parameters found 
        print("Best trial:")
        trial = study.best_trial

        print("Value: {:.4f}".format(trial.value))

        print("Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        params = {}
        #params.update(base_params)
        params.update(study.best_trial.params)
        
        if self.name == 'cat':
            model = catboost.CatBoostClassifier(**param)     
            model.fit( X_train, y_train, eval_set=[(self.X_val, self.y_val)], early_stopping_rounds=100 )
        elif self.name == 'rf':
            model = RandomForestClassifier(**param)     
            model.fit( X_train, y_train, eval_set=[(self.X_val, self.y_val)], early_stopping_rounds=100 )
        return model


# run as a script
if __name__ == "__main__":
  met = 'cat'
  my_opt = myOptuna( met, X_train, y_train, X_val, y_val )
  model = my_opt.run(n_trials=2)
  
  pref = f"{met}_offset{O}"
  print('saving to', pref)
  
  joblib.dump( model, pref + '.joblib', compress=3)  # compression is ON!
  joblib.dump( params,  pref + '_params.txt' )  # compression is ON!
