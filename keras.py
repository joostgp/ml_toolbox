# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 11:46:27 2016

WIP

@author: joostgp
"""

"""
Intented usage:
Use a single Keras callback class while training a Keras model to manage early 
stopping and properly storing the weights and metadata


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

import pandas as pd

from keras.callbacks import Callback

class prediction_check(Callback):
    def __init__(self, filename = './nn_model.h5', tag=''):
        self.predhist = pd.DataFrame(columns=['acc','logloss','val_acc','val_logloss'])
        self.tag = str(tag)
        self.counter = 0
        self.best = 10.0
        self.best_rounds = 1
        self.wait = 0
        self.patience = 3
        self.filename = filename
        
    def on_epoch_end(self, epoch, logs={}):
        #print logs.values()
        self.counter += 1
        self.predhist.loc[epoch] = logs.values()
        
        current_score = logs['val_loss']
        
        if current_score < self.best:
            self.best = current_score
            self.best_train = logs['loss']
            self.best_rounds = self.counter
            self.wait = 0
            self.model.save_weights(self.filepath, overwrite=True)
        else:
            if self.wait >= self.patience:
                self.model.stop_training = True
                self.model.load_weights(self.filepath)
                print('Best number of rounds: %d \n Val loss: %f \n' % (self.best_rounds, self.best))
            self.wait += 1
            
    def on_train_end(self, logs={}):
        # At the end of training load weights of best model and save results
        self.model.load_weights(self.filename)
        self.model.save(self.filepath, overwrite=True)