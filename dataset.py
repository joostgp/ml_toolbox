# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 19:12:44 2016

@author: joostbloom
"""

import os

from collections import namedtuple

import pandas as pd



class SourceDataSet:
    
    
    DataFiles = namedtuple('DataFiles','train test target')
    
    datafiles = DataFiles(train='filename_train.csv', 
                          test='filename_test.csv', 
                          target='filename_target.csv or column name train')
    base_path = 'base path'
    id_col = 'Id col'
    target_col = ['Target']
    
    sep = DataFiles(train = ',', test=',', target=';')
    
    def __init__(self, train=None, test=None, target=None):     
        
        if isinstance(train, str):
            self.datafiles = self.DataFiles(train=train, 
                                       test=test, 
                                       target=target)
        else:
            self.datafiles = self.datafiles
            
        self.traindata = None
        self.testdata = None
        self.alldata = None
        self.target = None
        self.cols = self.get_cols()
        self.n_train = None
        self.n_test = None
    
    def load(self, dataset='train'):
        
        datafile = self.full_path(getattr(self.datafiles, dataset))
        
        # Load train or test data
        setattr(self, dataset + 'data', pd.read_csv(datafile, 
                                                    index_col=0, 
                                                    sep=getattr(self.sep, dataset)))
        
        #return self
        
        # Load target data
        if dataset=='train':
            if self.datafiles.target in self.traindata.columns:
                # From train set if column exists
                self.target = self.traindata[self.datafiles.target]
                self.traindata.drop(self.datafiles.target, 
                               axis=1, 
                               inplace=True)
            elif os.path.exists(os.path.join(self.base_path,
                                             self.datafiles.target)):
                # Load from csv file if file exists
                self.target = self.load_target_from_file()
            
        return self
    
    def merge(self):
        self.alldata = pd.concat([self.traindata, self.testdata], axis=0)
        self.n_train = range(self.traindata.shape[0])
        self.n_test = range(self.traindata.shape[0], self.traindata.shape[0] + \
                                                         self.testdata.shape[0])
        
        return self
    
    def full_path(self, f):
        return os.path.join(self.base_path, f)
    
    def get_cols(self):
        if isinstance(self.traindata, pd.DataFrame):
            return self.traindata.columns
        if isinstance(self.testdata, pd.DataFrame):
            return self.testdata.columns
        else:
            return pd.read_csv(self.full_path(self.datafiles.train), 
                               nrows=0).columns
        
    def get_ids(self, dataset='train'):
        data = getattr(self, '{}data'.format(dataset))
        
        if isinstance(data, pd.DataFrame):
            return data.index.values
        else:
            return pd.read_csv(self.full_path(getattr(self.datafiles, dataset)), 
                               sep=getattr(self.sep, dataset),
                               usecols=[self.id_col])[self.id_col].values
    
    def load_target_from_file(self):
        return pd.read_csv(self.full_path(self.datafiles.target), 
                               sep=getattr(self.sep, 'target'),
                               usecols=[self.id_col] + self.target_col, 
                               index_col=0)
    
    def get_target(self):
        if not isinstance(self.target, pd.DataFrame):
            # Target is loaded with traindata, in case not loaded do it here
            self.target =  self.load_target_from_file()
        return self.target
        
    def describe(self):
        print('Datasets: {}'.format(self.datafiles))
        print('Train size: {}'.format(self.traindata.shape))
        print('Test size: {}'.format(self.testdata.shape))
        print('Target size: {}'.format(self.target.shape))
        print('Train head:')
        print(self.traindata.head(3))
        print('Test head:')
        print(self.testdata.head(3))
        print('Target head:')
        print(self.target.head(3))
        
    def __str__(self):
        if isinstance(self.data, pd.DataFrame):
            return 'Data from {}: {}'.format(self.datafile, 
                                                   self.data.shape)
        else:
            return 'Data from {}: (not loaded yet)'.format(self.datafiles)