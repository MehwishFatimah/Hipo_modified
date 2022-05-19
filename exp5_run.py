import os
import pandas as pd
#from pathlib import Path
import json
from hipo_rank.summarizers.lead import LeadSummarizer
from hipo_rank.summarizers.oracle import OracleSummarizer
from hipo_rank.dataset_iterators.pubmed import PubmedDataset
#from hipo_rank.evaluators.rouge import evaluate_rouge
from hipo_rank.evaluators.rouge_mf import Evaluate
import time
from tqdm import tqdm

"""
Lead
"""
DEBUG = False #True

# Parent Directory path
OUT_PATH = "/hits/basement/nlp/fatimamh/outputs/hipo/"
# Directory
DIR = "exp05/"

# Path
PATH = os.path.join(OUT_PATH, DIR)
if not os.path.exists(PATH):
    os.makedirs(PATH)

DATASETS = [
    ("wiki_cross_test", PubmedDataset, {"file_path": "/hits/basement/nlp/fatimamh/inputs/wiki_pub_style/cross/test.txt"}),
    ("wiki_cross_val", PubmedDataset, {"file_path": "/hits/basement/nlp/fatimamh/inputs/wiki_pub_style/cross/val.txt"}),
]
NUM_WORDS = {
    'wiki_cross_test': 200,
    'wiki_cross_val': 200
}
SUMMARIZERS = [
    ('oracle', OracleSummarizer, {}),
    ('lead', LeadSummarizer, {}),

]

for (dataset_id, dataset, dataset_args) in DATASETS:
    DataSet = dataset(**dataset_args)
    docs = list(DataSet)
    if DEBUG:
        docs = docs[:5]

    for (summarizer_id, summarizer, summarizer_args) in SUMMARIZERS:
        summarizer_args.update(dict(num_words=NUM_WORDS[dataset_id]))
        Summarizer = summarizer(**summarizer_args)
        folder = f"{dataset_id}_{summarizer_id}"
        experiment_path = os.path.join(PATH, folder)
        print(experiment_path)
        print('---------------------')
        #experiment_path = RESULTS_PATH / f"{dataset_id}_{summarizer_id}"
        try:
            #experiment_path.mkdir(parents=True)
            if not os.path.exists(experiment_path):
                os.makedirs(experiment_path)
            
            results = []
            references = []
            summaries = []
            df = pd.DataFrame()

            for doc in tqdm(docs):
                summary = Summarizer.get_summary(doc)
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

            #if os.path.exists(file):
            #    score = Evaluate()
            #   score.rouge_scores(file, out_file)      

            #rouge_result = evaluate_rouge(summaries, references)
        except FileExistsError:
            print(f"{experiment_path} already exists, skipping...")
            pass


