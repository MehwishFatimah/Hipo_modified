import os
import pandas as pd
from transformers import pipeline
from nltk import tokenize
import torch
from hipo_rank.evaluators.rouge_mf import Evaluate

from torch.utils.data import Dataset

device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))
translator = pipeline("translation_en_to_de", model="t5-base", device=0)



class MyDataset(Dataset):
    
    def __init__(self, text):

        self.sample = text
        
    def __getitem__(self, idx):
        return self.sample[idx]
    
    def __len__(self):
        return len(self.sample)

'''-------------------------------------------------------
'''
def translate(text: list):
    sent = tokenize.sent_tokenize(text)
    dataset = MyDataset(sent)
    
    trans = []
    for i in range(len(dataset)):
        if len(dataset[i]) <= 1:
            continue
        print('len(dataset[i]): {}'.format(len(dataset[i])))
        t = translator(dataset[i], max_length=len(dataset[i]))
        t = t.pop(0)['translation_text']
        #print("s: {}\nt: {}".format(s, t))
        trans.append(t)
    print("translated: {}\n".format(trans))
    return trans


'''-------------------------------------------------------
'''

if __name__ == "__main__":

    print("\n\n--------------------------------------------------\n\n")
    
    path = "/hits/basement/nlp/fatimamh/outputs/hipo/exp10/wiki_cross_test-st_bert_base-cos-edge"

    file = os.path.join(path, 'simple_summaries.csv')
    
    #print(file)
    df = pd.read_csv(file)
    #df=df.head(10)
    df['system']= df['system'] #.to_string()
    df['system']= df['system'].apply(lambda x: translate(x))
    print(df.head())

    file = os.path.join(path, 'simp_trans_summaries.csv')
    df.to_csv(file)
    print('file saved: {}'.format(file))
    
    out_file = os.path.join(path, 'simp_trans_scores.csv')
    if os.path.exists(file):
        score = Evaluate()
        score.rouge_scores(file, out_file)
    
    print("\n\n--------------------------------------------------\n\n") 

