#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:36:25 2022

@author: fatimamh
"""

import os
import pandas as pd
from nltk.tokenize import sent_tokenize
'''----------------------------------------------------------------
'''

def get_folders(root):
    folders = list(filter(lambda x: os.path.isdir(os.path.join(root, x)), os.listdir(root)))
    #print(folders)
    return folders


'''----------------------------------------------------------------
'''
def get_pairfolder(f):
    parts = f.split('_')
    print(parts)
    parts[1] = 'cross'
    print(parts)
    out_f = '_'.join(parts)
    return out_f

'''----------------------------------------------------------------
'''
if __name__ == '__main__':
    
    """ 
    open simple_summaries from mono folder. We have simplified summaries in it.
    Now we have to merge cross reference summaries with simplified so it can be used for scoring. 
    Also scoring for mono has to be done. Convert it into list.
    Doing only merging here. mono sentence split
    11-9 simple summaries for mono has been converted
    """
    root = "/hits/basement/nlp/fatimamh/outputs/hipo/exp06"
    folders = get_folders(root)
    subs = 'mono'
    file = 'summaries.csv'
    new_file = 'simple_summaries.csv'
    
    mono_df = pd.DataFrame()
    cross_df = pd.DataFrame()
    
    count = 0 
    for f in folders:
        if subs in f:
            count = count + 1
            mono_folder = os.path.join(root, f)
            cross_folder = get_pairfolder(f)
            cross_folder = os.path.join(root, cross_folder)
            
            if os.path.isdir(mono_folder):
                print("mono folder: {}".format(mono_folder))
                mono_file = os.path.join(mono_folder, new_file)
                if os.path.isfile(mono_file):
                    print("mono file: {}\n".format(mono_file))
                    mono_df = pd.read_csv(mono_file, index_col= False)
            
            if os.path.isdir(cross_folder):
                print("cross folder: {}".format(cross_folder))
                cross_file = os.path.join(cross_folder, file)
                if os.path.isfile(cross_file):
                    print("cross file: {}\n".format(cross_file))
                    cross_df = pd.read_csv(cross_file, index_col= False)
                    cross_df.drop(cross_df.columns[cross_df.columns.str.contains('unnamed',case = False)], 
                                                                      axis = 1, inplace = True)
                    if 'meta' in cross_df.columns:
                        cross_df.drop('meta', axis= 1, inplace=True)
           
            print('\n===========================================================================')
            print(mono_df.head(3))
            print(cross_df.head(3))
            print('\n===========================================================================')
            
            cross_df['system'] = mono_df['system']
            mono_df['system'] = mono_df['system'].apply(lambda x: sent_tokenize(x))
            
            print(mono_df.head(5))
            print(cross_df.head(5))
            print('\n===========================================================================')
            
            out_file = os.path.join(cross_folder, new_file)
            print("new cross file: {}\n".format(out_file))
            cross_df.to_csv(out_file, index=False)
            out_file = os.path.join(mono_folder, 'sim_summaries.csv')
            print("mono file: {}\n".format(out_file))
            mono_df.to_csv(out_file, index=False)
            print('total folders processed: {}'.format(count))
            
            
            
            
    
