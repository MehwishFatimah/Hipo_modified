import os
import pandas as pd
from transformers import pipeline
from nltk import tokenize
import torch
from hipo_rank.evaluators.rouge_mf import Evaluate

path = '/hits/basement/nlp/fatimamh/outputs/hipo/exp02/wiki_cross_test-pacsum_bert-cos-backloaded_edge-add_f=0.0_b=1.0_s=1.5'

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
    subs = "wiki_cross_test"
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

    translator = pipeline("translation_en_to_de", model="t5-base", device=0)
    #translator.to(device)
    

    file = os.path.join(path, 'summaries.csv')
    #print(file)
    df = pd.read_csv(file)
    #df=df.head(10)
    df['system']= df['system'].apply(lambda x: translate(x))
    print(df.head())

    df.to_csv(file)
    print('file saved: {}'.format(file))
    
    out_file = os.path.join(path, 'scores.csv')
    if os.path.exists(file):
        score = Evaluate()
        score.rouge_scores(file, out_file)
    print("\n\n--------------------------------------------------\n\n")    
    
            