# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 11:46:27 2016

WIP

@author: joostgp
"""

"""
MyKeras is a convenience class for easily evaluating Keras models using HyperOpt

Intented usage:
Use a single class to initiate keras models and doing hyperoptimization using
HyperOpt. This class stores progress and results and provides methods for 
evaluating the progress and results.


Blue print:
- Data to store
    - Model weights
    - Model architecture
    - Copy of script
    - Cross validation strategy
    - Features
    - Samples
    - Timestamp
- Input:
    - Output dir and filenames
    - Loss function type
- Call methods to implement:
    - On-epoch-end
    - On-train-end
"""
import random
import os


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import log_loss


from keras.callbacks import Callback


from hyperopt import fmin, tpe, STATUS_OK, Trials

class MyKeras(object):
    
    def __init__(self, 
                 params_range, model_def, 
                 train_data, val_data, 
                 description='',  save_dir='./'):
        
        self.params_range = params_range
        self.model_def = model_def
        self.description = description
        self.save_path = os.path.join(save_dir, 
                                      self.description + '_model.h5')
        self.train_data = train_data
        self.val_data = val_data
         
        # For hyperopt
        self.trials = Trials()  

        # Below are set after optimization
        self.best_params = None                         
        
    def nn_score(self, params):
        
        # Does not work yet for nested params
        #if not params:
        #    params = self.best_params
        # create model
        print ('Model with following parameters: %s' % (params))
        
        if 'seed' in params:
            np.random.seed(params['seed'])
            random.seed(params['seed'])
        
        model = self.model_def(params)
    
        monitor = prediction_monitor(self.save_path)
    
        model.fit(self.train_data[0].todense(), 
                  self.train_data[1], 
                  nb_epoch=int(params['n_epoch']), 
                  batch_size=int(params['batch_size']), 
                  validation_data=(self.val_data[0].todense(), self.val_data[1]), 
                  callbacks=[monitor], 
                  verbose=2) 
        
        # evaluate the model
        pred_val = model.predict_proba(self.val_data[0].todense(), 
                                       batch_size=params['batch_size'], 
                                       verbose=0)
    
        cv_score = log_loss(self.val_data[1], pred_val)
    
        print('logloss val {}'.format(cv_score))
        
        return cv_score, monitor
        
    def nn_score_hyperopt_wrapper(self, params):
        score = self.nn_score(params)    
        
        return {'loss': score[0], 'status': STATUS_OK}
        
    def hyperoptimize(self, max_evals=200):
        best = fmin(self.nn_score_hyperopt_wrapper, 
                    self.params_range, 
                    algo=tpe.suggest, 
                    max_evals=max_evals, 
                    trials=self.trials)
        
        # Update params
        self.best_params = best
        # Does not work for nested params
        #self.best_params = self.params_range
        #self.best_params.update(best)

    
class prediction_monitor(Callback):
    def __init__(self, filename = './nn_model_tmp.h5', tag=''):
        self.predhist = pd.DataFrame(columns=['acc','logloss','val_acc','val_logloss'])
        self.tag = str(tag)
        
        # For early stopping
        self.wait = 0
        self.patience = 3
        
        # Initial values
        self.best_score = 99999.0
        self.best_rounds = 1
        
        # Path and filename
        self.filename = filename
        
    def on_epoch_end(self, epoch, logs={}):
        """ Called after each epoch. Keeos track of best score. """
        
        # Store progress in dataframe
        self.predhist.loc[epoch] = logs.values()
        
        # Check new score against best score
        new_score = logs['val_loss']
        
        if new_score < self.best_score:
            self.best_score = new_score
            self.best_rounds = epoch
            self.wait = 0
            self.model.save_weights(self.filename, overwrite=True)
        else:
            
            if self.wait >= self.patience:
                self.model.stop_training = True
                self.model.load_weights(self.filename)

            self.wait += 1
            
    def on_train_end(self, logs={}):
        """ Called after each training. Loads weights of best epoch."""
        
        self.model.load_weights(self.filename)
        self.model.save(self.filename, overwrite=True)
        
        print('Best score %f after %d epochs.' \
              % (self.best_rounds, self.best_score))
              
    def plot_loss_progress(self, ax1=None, ax2=None, c='b'):
        """ Plots values of loss function during optimization """        
        
        
        if not ax1:
            fig, (ax1, ax2) = plt.subplot(1, 2, figsize=(16,4))
        
        self.predhist.plot(y='acc', ax=ax1, 
                           label=self.tag, color=c)
        self.predhist.plot(y='val_acc', ax=ax1, 
                           label=self.tag, linestyle='--', color=c)
        ax1.set_title('Accuracy score')
        ax1.set_xlabel('Epoch #')
        ax1.set_ylabel('Accuracy')
        
        self.predhist.plot(y='logloss', ax=ax2, 
                           label=self.tag, color=c)
        self.predhist.plot(y='val_logloss', ax=ax2, 
                           label=self.tag, linestyle='--', color=c)
                           
        ax2.set_title('Logloss score')
        ax2.set_xlabel('Epoch #')
        ax2.set_ylabel('Logloss')
        
