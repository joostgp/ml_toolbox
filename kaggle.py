# -*- coding: utf-8 -*-
""" This module provides a class for automatic creation of Kaggle
    submission file.
    
    

    Usage:
    # this creates a submission file 
    kag = KaggleResult(index=..., data=...)
    
    # check correct dimensions of data and upload
    if kag.validate()
        kag.upload()
    
    # Leaderboard score
    print kag.lb_score
        
    # Optionally a description can be added
    
    Requirements:
    - Pandas
    - Numpy
    - Mechanize
    
    Two configuration files:
    in user home directory: .kag_account:
    
    [Kaggle_Account_Info]
    kag_username = hidden
    kag_password = hidden
    
    in Kaggle project directory .kag_competition:
    
    [competition]
    name=talking_data
    login_url=https://www.kaggle.com/account/login
    upload_url=https://www.kaggle.com/c/talkingdata-mobile-user-demographics/submissions/attach
    submission_url=https://www.kaggle.com/c/talkingdata-mobile-user-demographics/submissions
    sample_submission=data_ori/sample_submission.csv
    
    [settings]
    maxtime=180
    repeat=20
    
Created on Tue Jul 12 11:59:32 2016

@author: joostgp
"""

import datetime
import os
import time
import cookielib

from mechanize import Browser 
from ConfigParser import SafeConfigParser

import pandas as pd
import numpy as np
  
  
class KaggleError(Exception):
    """ Error thrown by KaggleResult. """
    def __init__(self, message):
        self.message = message
        print message
        
    def __str__(self):
        return repr(self.message)


class KaggleResult(object):

    """ Class for creating and uploading Kaggle submission files. """
    
    kag_account_config_file = os.path.join(os.path.expanduser("~"), '.kag_account')
    kag_competition_info = os.path.join(os.getcwd(), '.kag_competition')
    
    def __init__(self, data, index=None, cv_score=-1, 
                 description='', subdir = '', verbose=False):
        """ KaggleResult file used to create submission file and upload them
        to the Kaggle leaderboard
        
        Parameters:
        -----------
        data: numpy array or pandas dataframe
            numpy array containing the predictions
        index: list / 1D numpy array (optional, default None)
            values of the index of each prediction row. Will be first column of
            submission file. Can be included with data.
        cv_score: float (optional, default-1)
            score achieved locally
        description: string (optional, default '')
            description of this submission file. Will be used on the Kaggle and
            in a log file
        subdir: string (optional, default '')
            the submission file will be create in this subdirectory of working
            directory. It will be created if it does not exist
        verbose: bool (optional, default False)
            If true, it will print progress reports on console
        
        Returns:
        lb_score: float
            float with the leaderboard score. -1 is score could not be found.
        """ 
        
        # Check for configuration files
        if not os.path.isfile(self.kag_account_config_file):
            raise KaggleError('Kaggle account info not found in {}' \
                                .format(self.kag_account_config_file))
                                
        if not os.path.isfile(self.kag_competition_info):
            raise KaggleError('Kaggle competition info not found in {}' \
                                .format(self.kag_competition_info))
                                
        # Load configuration data
        self.load_competition_config()
        self.load_kaggle_account_config()
            
        
        if isinstance(data, np.ndarray):
            if isinstance(index, np.ndarray):
                self.data = pd.DataFrame(data, index = index).reset_index()
            else:
                self.data = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            if data.shape[1]==len(self.get_columns())-1:
                self.data = data.reset_index()
            else:
                self.data = data
        else:
            raise ValueError('data or index should be np.ndarray or pd.DataFrame')
        
        # Make sure columns are correct
        self.data.columns = self.get_columns()
            
        now = datetime.datetime.now()
            
        self.sub_path = os.path.join(os.getcwd(),subdir)
        self.cv_score = cv_score
        self.lb_score = -1
        self.description = description
        self.verbose = verbose
        self.timestamp = now
    
        # Create subdir if does not exist
        if not os.path.isdir(self.sub_path):
            os.makedirs(self.sub_path)
            
        # Create submission file
        if data is not None:
            self.create_submission_file()
    
    def load_competition_config(self):
        """ Loads Kaggle competition info from config file """
        try:
            parser = SafeConfigParser()
            parser.read(self.kag_competition_info)
            
            self.kag_name = parser.get('competition', 'name')
            self.kag_login_url = parser.get('competition', 'login_url')
            self.kag_upload_url = parser.get('competition', 'upload_url')
            self.kag_submissions_url = parser.get('competition', 'submission_url')
            self.sample_submission = parser.get('competition', 'sample_submission')
            
            self.maxtime = parser.getint('settings', 'maxtime')
            self.repeat = parser.getint('settings', 'repeat')
        except:
            raise KaggleError('Kaggle competition info could not be read from {}' \
                                .format(self.kag_competition_info)) 
        

    def load_kaggle_account_config(self):
        """ Loads Kaggle account info from config file """
        try:
            parser = SafeConfigParser()
            parser.read(self.kag_account_config_file)
            
            self.kag_username = parser.get('Kaggle_Account_Info', 'kag_username')
            self.kag_password = parser.get('Kaggle_Account_Info', 'kag_password')
        except:
            raise KaggleError('Kaggle account info could not be read from {}' \
                                .format(self.kag_account_config_file))    
        

    def create_submission_file(self):
        """ Create submission file and log file with description
        Parameters:
        -----------
        None
         
        Returns:
        filename: string
            a string with the full path to submission file
        """
        if not isinstance(self.data, pd.DataFrame):
            raise KaggleError('No data loaded')
        
        # Create submission file
        sub_file = self.get_file_name() 
        
        self.data.to_csv(self.get_file_path(), index=False)
#        with open( os.path.join(self.sub_path, sub_file),'w') as f:
#            f.write(','.join(self.hdlist) + '\n')
#            
#            for i in range(len(self.index)):
#                s = str(self.index[i])
#                s += ',' + ','.join(self.data[i].astype(str)) + '\n'
#                f.write(s)
        
        # Create logfile
        if self.description:
            logfile = self.get_file_name('log') 
            with open(os.path.join(self.sub_path, logfile),'w') as f:
                f.write('Description: {}'.format(self.description))
        
        return os.path.join(self.sub_path, sub_file)
    
    def get_file_path(self, filetype='csv'):
        return os.path.join(self.sub_path, self.get_file_name(filetype) )
    
    def get_file_name(self, filetype='csv'):
        """ Get file name of submission file
        Parameters:
        -----------
        type: str
            string indicating type of fyle. 'csv' for submission file and 
                'log' for logfile
        
        Returns:
        filename: str
            a string with the submission file name without extension
        """
        
        return 'submission_{:.4f}_{}_{:.4f}.{}'.format(self.cv_score, 
                    self.timestamp.strftime("%Y-%m-%d-%H-%M"), self.lb_score, filetype)
    
    def get_data(self):
        """ Return data in submission file
        Parameters:
        -----------
        None
        
        Returns:
        filename: pd.DataFrame
            a pandas DataFrame with all data
        """
        return pd.read_csv(self.get_file_path())
        
    def get_columns(self):
        """ Get columns based on sample submission file """
        
        return pd.read_csv(self.sample_submission).columns

    def validate(self):
        """ Do a quick validation on structure of dataframe
        Parameters:
        -----------
        None
        
        Returns:
        result: tuple (bool, string)
            tuple with first element is true when file is correct and second
                element is error string
        """
        
        # Compare to sample submission file 
        sample = pd.read_csv(self.sample_submission)
        
        msg = None        

        if self.data.shape != sample.shape:
            msg = 'data in correct shape {} vs {}'.format(self.data.shape, 
                                                         sample.shape)
        
        if self.data.isnull().values.any():
            msg = 'data contains missing values'
            
        if not np.all([self.data.columns, sample.columns]):
            msg = 'wrong columns headers'
        
        if msg:
            return (False, msg)
        else:
            return (True, 'all_ok')

    def login_to_kaggle(self):  
        """ Login to Kaggle website
        Parameters:
        -----------
        None
        
        Returns:
        browser: Browser
            a mechanizer Browser object to be used for further access to site
        """          
        
        if self.verbose:
            print("Logging in to Kaggle..."),

        br = Browser()
        cj = cookielib.LWPCookieJar()
        br.set_cookiejar(cj)
        
        br.open(self.kag_login_url)
        
        br.select_form(nr=0)
        br['UserName'] = self.kag_username
        br['Password'] = self.kag_password
        br.submit(nr=0)
        
        if br.title() == "Login | Kaggle":
            raise KaggleError("Unable to login Kaggle with username %s (response title: %s)" % (self.kag_username,br.title()))
        
        if self.verbose:
            print("done!")
        
        return br
    
    def upload(self):
        """ Upload submission file to Kaggle leaderboard
        Parameters:
        -----------
        None
        
        Returns:
        lb_score: float
            float with the leaderboard score. -1 is score could not be found.
        """  
        
        subfilepath = os.path.join(self.sub_path, self.get_file_name())
        
        if not os.path.isfile(subfilepath): 
            raise KaggleError("Submission file %s not found" % subfilepath)
        
        br = self.login_to_kaggle()     
        
        if self.verbose:
            print("Uploading %s..." % subfilepath),   
        
        br.open(self.kag_upload_url)
        
        r = br.response().get_data()
        #print r
        ss = "Your team has used its submission allowance (5 of 5). This resets at midnight UTC ("
        i_s = r.find(ss)
        if i_s>-1:
            i_e = r.rfind(" from now).")
            raise KaggleError("Submission limit reached, please wait %s." % r[i_s+len(ss):i_e])
 
        br.select_form(nr=0)
        
        br.add_file(open(subfilepath), 'application/octet-stream',
                    os.path.basename(subfilepath), name='SubmissionUpload')
                    
        br['SubmissionDescription'] = self.description
        br.submit(nr=0)
        
        if self.verbose:
            print("done!")
        
        return self.get_lb_score(br)
  
    
    def get_lb_score(self, br=None):
        """ Scrape leaderboard score for this submission file
        Parameters:
        -----------
        br: Browser object
            mechanizer browser object. If None will attempt to login to Kaggle.
        
        Returns:
        lb_score: float
            float with the leaderboard score. -1 is score could not be found.
        """  
        
        if br is None:        
            br = self.login_to_kaggle()
        
        s = time.time()
        subfilepath = self.get_file_path()
        logfilepath = self.get_file_path('log')
        
        success = False
        
        if self.verbose:
            print("Waiting for score %s..." % subfilepath),
        
        while time.time()-s < self.maxtime:
            
            
            # Open page with submission and get HTML
            br.open(self.kag_submissions_url)
            html = br.response().get_data()
            
            # HTML Response looks something like this
            '''
            ...<td>
                <a class="file" href="/submissions/3257683/3257683.zip">sample_submission.csv</a>
            </td>
            <td class="center">2.48491</td>
            <td>
            <input class="submission-check" type="checkbox" name="checkedSubmission" value="3257683"/>...
            
            '''
            
            # Scrape table row with the name of this submissionfile and split per cell
            i_s = html.index(self.get_file_name())
            i_e = html[i_s:].index('</tr>')
            tablecells = html[i_s:i_s+i_e].split('</td>')
            
            # Get Score from this small piece of html in the second cell
            i_s = tablecells[1].index(">")+1
            score_str = tablecells[1][i_s:]
            
            # Lets see if it worked (if we can convert score to float we assume it is)
            try:
                score = float(score_str)
                success = True
                break
            except:
                if self.verbose:
                    print("try again in %ds..." % (self.repeat)),
            
            time.sleep(self.repeat)
        
        if success:
              
            self.lb_score = score
            
            # Rename submission file and logfile
            os.rename(subfilepath, os.path.join(self.sub_path, self.get_file_name()))
            
            if self.description:
                os.rename(logfilepath, os.path.join(self.sub_path, self.get_file_name('log')))
            
            if self.verbose:
                print("done!")
                
            return score
        else:
            if self.verbose:
                print "Could not get score from Kaggle in following websnippet: %s" % tablecells[1]
            return -1.0
    
        
