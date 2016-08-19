# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 17:22:54 2016

WIP
"""

"""
Intented usage:
Train a set of classifiers with varying parameters and input data and compare the results later in a consistent and efficient way.
- output should be stored in a directory
- directory contains models and metadata
- optionally automatically Kaggle submission files are created and uploaded to the leaderboard

Public methods:
- init with groupname and directory (should create dirs if not exists)
- add case
- run cases

Private methods:
- store results
- run case
- plots

"""

import os
import time
import json
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb 

from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

from kaggle import KaggleResult, KaggleError

class CaseRunner(object):
    
    def __init__(self, groupname, output_dir):
        self.groupname = groupname
        self.output_dir = os.path.join(os.getcwd(),output_dir,groupname)
        self.cases = []
    
        self.plot_cv_curves = False
        self.plot_feature_importance = False
        self.submit_to_kaggle = False
        
        self.scores = []
        self.score_plot_ylim = [2.2,2.6] 
        
        # TO-DO: Check and create directories
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        
    def add_case(self, casename, X, y, classifier, params, testsize = 0.2, random_state = 123, X_test=None, ids_test=None):
        self.cases.append( (casename,  X, y, classifier, params, testsize, random_state, X_test, ids_test) )
    
    def run_cases(self):
        print("Running %s" % self.groupname)    
        print("")
        
        clfs = []
        times_t = []
        times_p = []
        
        for c in self.cases:
            
            casename = c[0]
            classifier = c[3]
            params = c[4]
            testsize = c[5]
            X_test = c[7]
            
            tag = self.groupname + " - " + casename
            
            s = time.time()      
            
            # Train model
            (clf,df_eval, X_val, y_val) = self.run_case(c)
            clfs.append( clf )
            times_t.append( time.time()-s )
            
            if plot_cv_curves:
                plot_cv_curves(df_eval, self.groupname + " - " + casename, self.output_dir)
            
            # Calculate score
            s = time.time()  
            if classifier=="xgb":
                score = report_result(clf, X_val, y_val, params['booster']=='gbtree')
                self.scores.append( score )
            else:
                self.scores.append( report_result(clf, X_val, y_val) )
            times_p.append( time.time()-s )
            
            # Create submissions file if X_test is provided
            if X_test is not None:
                ids_test = c[8]
                lb_score = self.create_submission_file(clf, X_test, ids_test, score, tag)
            else:
                lb_score = -1
                
                
            # Save data and parameters
            params['test_size'] = testsize
            
            paramfile = os.path.join(self.output_dir, '%s_params.txt' % tag)
            json.dump(params, open(paramfile,'w'))
            
            # Save model
            modelfile = os.path.join(self.output_dir, '%s_xgbmodel.model' % tag)
            clf.save_model(modelfile)
            
            # Save features
            feat = pd.Series(clf.get_fscore()).sort_values(ascending=False)
            feature_file = os.path.join(self.output_dir, '%s_feat.txt' % tag)
            self.create_feature_file(feat, feature_file)
            
            if self.plot_feature_importance:
                self.plot_feature_importance(feat, tag)
            
            # Write to logfile
            logfile = os.path.join(self.output_dir, 'logfile.txt')
            if not os.path.exists(logfile):
                f = open(logfile,'w')
                f.write('Date,Groupname,Casename,Local score,LB Score,param,model,features\n')
                f.close()
            
            f = open(logfile,'a')
            f.write( str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) + ',' + \
                    self.groupname + ',' + \
                    casename + ',' + \
                    str(score) + ',' + \
                    str(lb_score) + ',' + \
                    paramfile + ',' + \
                    modelfile + ',' + \
                    feature_file + ',' + \
                    '\n')
            f.close()
            
        self.plot_group_scores()
        
                
    def run_case(self,case):
        name = case[0]
        X = case[1]
        y = case[2]
        classifier = case[3]
        clfparams = case[4]
        testsize = case[5]
        rs = case[6]
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=testsize, random_state=rs, stratify=y)
        
        print("Running case: %s" % name)
        
        if classifier=="xgb":
            (gbm, eval_result) = get_xgboost_classifier(X_train, y_train, X_val, y_val, clfparams, output_eval=True)
            
        return (gbm, eval_result, X_val, y_val)
        
    def plot_group_scores(self):
        width = 0.5    
        
        xlabels = [x[0] for x in self.cases]
    
        plt.figure()
        plt.title(self.groupname)
        plt.bar(np.arange(len(self.cases)),self.scores, width)
        plt.xticks(np.arange(len(self.cases))+width/2, xlabels, rotation='vertical')
        plt.ylabel("mllogloss Score")
        plt.ylim(self.score_plot_ylim)
        plt.xlim([-width,len(self.cases)])
        plt.tight_layout()
        plt.grid()
        
        ax = plt.gca()
        rects = ax.patches
        for rect, label in zip(rects, self.scores):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, height, "{:.4f}".format(label), ha='center', va='bottom')
            
        plt.savefig(os.path.join(self.output_dir,'%s_logloss.png' % self.groupname))
   
    def plot_feature_importance(self,df_features, tag):
        
        plt.figure()
        plt.title('Feature importance')
        df_features.plot(kind='bar')
        plt.tight_layout()
        plt.grid()
        plt.ylabel('Importance')
        plt.savefig(os.path.join(self.output_dir,'%s_feature_importance.png' % tag))
    
    def create_feature_file(self,df_features, featurefile):
        df_features.to_csv(featurefile)
     
    def create_submission_file(self, clf, X_test, ids_test, score, description):
    
        y_test = clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit)
        
        kag = KaggleResult(ids_test, prediction=y_test, score=score, description=description, sub_path=self.output_dir)
        
        if self.submit_to_kaggle:
            if kag.validate():
                try:
                    return kag.upload(description)
                except Exception as e:
                    print('Error uploading kaggle file:', e)
                    return -999
            else:
                print('Validation error submission file!!')
                return -99
        else:
            return -1



def get_xgboost_classifier(X_train, y_train, X_val, y_val, params, rs=123, output_eval=False):    
    
    if "objective" not in params: params["objective"] = "multi:softprob"
    if "booster" not in params: params["booster"] = "gbtree"
    if "eval_metric" not in params: params["eval_metric"] =  "mlogloss"
    
    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_val, y_val)
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    evals_result = {}
    gbm = xgb.train(params, dtrain, params['n_estimators'], evals=watchlist, early_stopping_rounds=20, verbose_eval=True, evals_result=evals_result)
    
    
    if output_eval:
        test_scores = evals_result['eval'][params["eval_metric"]]
        train_scores = evals_result['train'][params["eval_metric"]]
        
        df = pd.DataFrame()
        
        df['Eval set'] = test_scores
        df['Train set'] = train_scores
        
        return gbm, df
    else:
        return gbm




 

       
def report_result(clf, X_test, y_true, treebased=True):
    if str(type(clf)) == "<class 'xgboost.core.Booster'>":
        if treebased:   
            y_pred = clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit)  
        else:
            y_pred = clf.predict(xgb.DMatrix(X_test))  
    
    return log_loss(y_true, y_pred)
 

   
def plot_cv_curves(df, identifier, path):
    plt.figure()
    plt.title(str(identifier))
    plt.plot(df['Eval set'],'g',label='Validation set')
    plt.plot(df['Train set'],'r',label='Train set')
    plt.grid()
    plt.xlabel('Boosting round')
    plt.ylabel('Logloss Score')
    plt.legend()  
    plt.savefig(os.path.join(path, str(identifier)+'_eval_curves.png'))           

        
    
    
    
