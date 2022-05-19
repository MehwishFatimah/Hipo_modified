import os
import pandas as pd
from transformers import pipeline
from nltk import tokenize
import torch
from hipo_rank.evaluators.rouge_mf import Evaluate

rootdir = '/hits/basement/nlp/fatimamh/outputs/hipo/exp11'

count = 1
'''-------------------------------------------------------
'''
def translate(text: list):
    #print(text)
    sent = tokenize.sent_tokenize(text)
    #print(sent)
    #print()
    global count
    print('count: {}'.format(count))
    trans = []
    for s in sent:
        if len(s) <= 1:
            continue
        print('len(s): {}'.format(len(s)))
        t = translator(s, max_length=len(s))
        t = t.pop(0)['translation_text']
        #print("s: {}\nt: {}".format(s, t))
        trans.append(t)
    #print("translated: {}\n".format(trans))
    count = count+1
    
    return trans


'''-------------------------------------------------------
'''

def find_dirs():

    dir_list = []
    subs = "wiki_mono"
    #print("root: {}".format(rootdir))

    for path in os.listdir(rootdir):
        #print("path: {}\ntype: {}\n".format(path, type(path)))
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

    translator = pipeline("translation_en_to_de", model="t5-base", device=0)
    #translator.to(device)
    dirs = find_dirs()
    for path in dirs:

        #file = os.path.join(path, 'summaries_p1.csv')
        file = os.path.join(path, 'simple_summaries.csv')
        #print(file)
        df = pd.read_csv(file)
        #df=df.head(10)
        """
        df['system']= df['system'].to_string()
        df['system']= df['system'].apply(lambda x: translate(x))
        print(df.head())
        
        file = os.path.join(path, 'simp_trans_summaries.csv')
        df.to_csv(file)
        print('file saved: {}'.format(file))
        """
        out_file = os.path.join(path, 'sim_scores.csv')
        #out_file = os.path.join(path, 'scores_p1.csv')
        if os.path.exists(file):
            score = Evaluate()
            score.rouge_scores(file, out_file)
        print("\n\n--------------------------------------------------\n\n")    
    
            