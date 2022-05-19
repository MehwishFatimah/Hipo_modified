#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:36:25 2022

@author: fatimamh
"""

import os
import pandas as pd
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
    """
    root = "/hits/basement/nlp/fatimamh/outputs/hipo/exp11"
    folders = get_folders(root)
    subs = 'mono'
    file = 'summaries.csv'
    new_file = 'simple_summaries.csv'
    
    mono_df = pd.DataFrame()
    cross_df = pd.DataFrame()
    
    for f in folders:
        if subs in f:
            mono_folder = os.path.join(root, f)
            
            cross_folder = get_pairfolder(f)
            cross_folder = os.path.join(root, cross_folder)
            print(cross_folder)
            
            if os.path.isdir(mono_folder):
                print('Mono Directory exists')
                mono_file = os.path.join(mono_folder, new_file)
                if os.path.isfile(mono_file):
                    print(mono_file)
            
            if os.path.isdir(cross_folder):
                print('Cross Directory exists')
                cross_file = os.path.join(cross_folder, file)
                if os.path.isfile(cross_file):
                    print(cross_file)

          
        
