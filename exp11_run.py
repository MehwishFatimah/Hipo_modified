
import os
import pandas as pd
#from pathlib import Path
import json
from hipo_rank.summarizers.textrank import TextRankSummarizer
from hipo_rank.dataset_iterators.pubmed import PubmedDataset
from hipo_rank.evaluators.rouge_mf import Evaluate

import time
from tqdm import tqdm

"""
Textrank
"""
DEBUG = True

# Parent Directory path
OUT_PATH = "/hits/basement/nlp/fatimamh/outputs/hipo/"
# Directory
DIR = "exp11/"

# Path
PATH = os.path.join(OUT_PATH, DIR)
if not os.path.exists(PATH):
    os.makedirs(PATH)


DATASETS = [
    ("pubmed_test", PubmedDataset, {"file_path": "/hits/basement/nlp/fatimamh/inputs/pubmed-dataset/test.txt"}), # modified: data/pubmed-release
    ("arxiv_test", PubmedDataset, {"file_path": "/hits/basement/nlp/fatimamh/inputs/arxiv-dataset/test.txt"}), #data/arxiv-release
]
NUM_WORDS = {
    'pubmed_test': 200,
    'arxiv_test': 220
}
SUMMARIZERS = [
    ('textrank', TextRankSummarizer, {}),

]

for (dataset_id, dataset, dataset_args) in DATASETS:
    DataSet = dataset(**dataset_args)
    docs = list(DataSet)
    #print('docs: {}'.format(docs))
    if DEBUG:
        docs = docs[:5]
        print('docs len: {}'.format(len(docs)))
    for (summarizer_id, summarizer, summarizer_args) in SUMMARIZERS:
        
        summarizer_args.update(dict(num_words=NUM_WORDS[dataset_id]))
        print('summarizer_args: {}'.format(summarizer_args))

        Summarizer = summarizer(**summarizer_args)
        folder = f"{dataset_id}_{summarizer_id}"
        experiment_path = os.path.join(PATH, folder)
        print(experiment_path)
        print('---------------------')
        
        try:
            if not os.path.exists(experiment_path):
                os.makedirs(experiment_path)
            
            results = []
            references = []
            summaries = []
            df = pd.DataFrame()

            for doc in tqdm(docs):
                summary = Summarizer.get_summary(doc)
                #print('summary: {}'.format(summary))
                results.append({
                    "num_sects": len(doc.sections),
                    "num_sents": sum([len(s.sentences) for s in doc.sections]),
                    "summary": summary,

                })
                summaries.append([s[0] for s in summary])
                print('summaries len: {}'.format(len(summaries)))

                references.append(doc.reference)
                print('references len: {}'.format(len(references)))                
            
            df['reference'] = references
            df['system'] = summaries
            df['meta'] = results 

            file = os.path.join(experiment_path, 'summaries.csv')
            out_file = os.path.join(experiment_path, 'scores.csv')
            df.to_csv(file, encoding='utf-8')

            if os.path.exists(file):
                score = Evaluate()
                score.rouge_scores(file, out_file)      

            #rouge_result = evaluate_rouge(summaries, references)

        except FileExistsError:
            print(f"{experiment_path} already exists, skipping...")
            pass


