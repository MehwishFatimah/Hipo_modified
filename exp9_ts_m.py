import os
import pandas as pd
from transformers import pipeline
from nltk import tokenize
import torch
from hipo_rank.evaluators.rouge_mf import Evaluate

rootdir = '/hits/basement/nlp/fatimamh/outputs/hipo/exp09'

'''-------------------------------------------------------
'''
def translate(text: list):
    #print(text)
    sent = tokenize.sent_tokenize(text)
    #print(sent)
    #print()
    
    trans = []
    for s in sent:
        if len(s) <= 1:
            continue
        print('len(s): {}'.format(len(s)))
        t = translator(s, max_length=len(s))
        t = t.pop(0)['translation_text']
        #print("s: {}\nt: {}".format(s, t))
        trans.append(t)
    print("translated: {}\n".format(trans))
    return trans


'''-------------------------------------------------------
'''

def find_dirs():

    dir_list = []
    subs = "wiki_mono"
    #print("root: {}".format(rootdir))

    for path in os.listdir(rootdir):
        print("path: {}\ntype: {}\n".format(path, type(path)))
        path = os.path.join(rootdir, path)

        if os.path.isdir(path):
          
            if subs in path:
                dir_list.append(path)
    
    print(dir_list)
    
    return dir_list

'''-------------------------------------------------------
'''

if __name__ == "__main__":

    print("\n\n--------------------------------------------------\n\n")
    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    dirs = find_dirs()
    for path in dirs:

        file = os.path.join(path, 'simple_summaries.csv')
        #print(file)
        df = pd.read_csv(file)
       
        out_file = os.path.join(path, 'simple_scores.csv')
        if os.path.exists(file):
            score = Evaluate()
            score.rouge_scores(file, out_file)
        print("\n\n--------------------------------------------------\n\n")    
    
            