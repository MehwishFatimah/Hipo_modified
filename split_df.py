#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 17:54:22 2022

@author: fatimamh
"""
import os
import pandas as pd
import numpy as np

path = "/hits/basement/nlp/fatimamh/outputs/hipo/exp11/wiki_cross_test_textrank"


def split_dataframe_by_position(df, splits):
    """
    Takes a dataframe and an integer of the number of splits to create.
    Returns a list of dataframes.
    """
    dataframes = []
    index_to_split = len(df) // splits
    #print(index_to_split)
    start = 0
    end = index_to_split
    for split in range(splits):
        temporary_df = df.iloc[start:end, :]
        dataframes.append(temporary_df)
        start += index_to_split
        print(start)
        end += index_to_split
        print(end)
        print()
    return dataframes



'''----------------------------------------------------------------
'''
if __name__ == '__main__':
    
    #Do work here
    file = os.path.join(path, "summaries.csv")
    df = pd.read_csv(file, index_col= False)
    df.drop(df.columns[df.columns.str.contains('unnamed',case = False)], axis = 1, inplace = True)
    split_dataframes = split_dataframe_by_position(df, 2)
    
    print(len(split_dataframes[1]))
    