from hipo_rank.dataset_iterators.pubmed import PubmedDataset

from hipo_rank.embedders.w2v import W2VEmbedder
from hipo_rank.embedders.rand import RandEmbedder
from hipo_rank.embedders.bert import BertEmbedder
from hipo_rank.embedders.sent_transformers import SentTransformersEmbedder

from hipo_rank.similarities.cos import CosSimilarity

from hipo_rank.directions.undirected import Undirected
from hipo_rank.directions.order import OrderBased
from hipo_rank.directions.edge import EdgeBased

from hipo_rank.scorers.add import AddScorer
from hipo_rank.scorers.multiply import MultiplyScorer

from hipo_rank.summarizers.default import DefaultSummarizer
#from hipo_rank.evaluators.rouge import evaluate_rouge
from hipo_rank.evaluators.rouge_mf import Evaluate

#from pathlib import Path
import os
import pandas as pd

import json
import time
from tqdm import tqdm

DEBUG = True

# PubMed hyperparameter gridsearch and ablation study

# Parent Directory path
OUT_PATH = "/hits/basement/nlp/fatimamh/outputs/hipo/"
# Directory
DIR = "exp01/"

# Path
PATH = os.path.join(OUT_PATH, DIR)
if not os.path.exists(PATH):
    os.makedirs(PATH)

DATASETS = [
    ("pubmed_val", PubmedDataset, {"file_path": "/hits/basement/nlp/fatimamh/inputs/pubmed-dataset/val.txt"}),
    ("pubmed_val_no_sections", PubmedDataset,
     {"file_path": "/hits/basement/nlp/fatimamh/inputs/pubmed-dataset/val.txt", "no_sections": True}
     ),
]
EMBEDDERS = [
    #("rand_200", RandEmbedder, {"dim": 200}),
    #("biomed_w2v", W2VEmbedder,{"bin_path": "/hits/basement/nlp/fatimamh/codes/HipoRank-master/models/wikipedia-pubmed-and-PMC-w2v.bin"}),
    ("biobert", BertEmbedder,
     {"bert_config_path": "/hits/basement/nlp/fatimamh/codes/HipoRank-master/models/biobert_v1.1_pubmed/bert_config.json",
      "bert_model_path": "/hits/basement/nlp/fatimamh/codes/HipoRank-master/models/biobert_v1.1_pubmed/pytorch_model.bin",
      "bert_tokenizer": "bert-base-cased"}
     ),
    ("bert", BertEmbedder,
     {"bert_config_path": "",
      "bert_model_path": "",
      "bert_tokenizer": "bert-base-cased",
      "bert_pretrained": "bert-base-cased"}
     ),
    ("pacsum_bert", BertEmbedder,
     {"bert_config_path": "/hits/basement/nlp/fatimamh/codes/HipoRank-master/models/pacssum_models/bert_config.json",
      "bert_model_path": "/hits/basement/nlp/fatimamh/codes/HipoRank-master/models/pacssum_models/pytorch_model_finetuned.bin",
      "bert_tokenizer": "bert-base-uncased",
      }
    ),
    ("st_bert_base", SentTransformersEmbedder,
         {"model": "bert-base-nli-mean-tokens"}
        ),
    #("st_roberta_large", SentTransformersEmbedder,
    #     {"model": "roberta-large-nli-mean-tokens"}
    #   ),
]
SIMILARITIES = [
    ("cos", CosSimilarity, {}),
]
DIRECTIONS = [
    ("undirected", Undirected, {}),
    ("order", OrderBased, {}),
    ("edge", EdgeBased, {}),
    ("backloaded_edge", EdgeBased, {"u": 0.8}),
    ("frontloaded_edge", EdgeBased, {"u": 1.2}),
]

SCORERS = [
    ("add_f=0.0_b=1.0_s=1.0", AddScorer, {}),
    ("add_f=0.0_b=1.0_s=1.5", AddScorer, {"section_weight": 1.5}),
    ("add_f=0.0_b=1.0_s=0.5", AddScorer, {"section_weight": 0.5}),
    ("add_f=-0.2_b=1.0_s=1.0", AddScorer, {"forward_weight":-0.2}),
    ("add_f=-0.2_b=1.0_s=1.5", AddScorer, {"forward_weight":-0.2, "section_weight": 1.5}),
    ("add_f=-0.2_b=1.0_s=0.5", AddScorer, {"forward_weight":-0.2,"section_weight": 0.5}),
    ("add_f=0.5_b=1.0_s=1.0", AddScorer, {"forward_weight":0.5}),
    ("add_f=0.5_b=1.0_s=1.5", AddScorer, {"forward_weight":0.5, "section_weight": 1.5}),
    ("add_f=0.5_b=1.0_s=0.5", AddScorer, {"forward_weight":0.5,"section_weight": 0.5}),
    ("multiply", MultiplyScorer, {}),
]


Summarizer = DefaultSummarizer()

experiment_time = int(time.time())


for embedder_id, embedder, embedder_args in EMBEDDERS:
    Embedder = embedder(**embedder_args)
    for dataset_id, dataset, dataset_args in DATASETS:
        DataSet = dataset(**dataset_args)
        docs = list(DataSet)
        if DEBUG:
            docs = docs[:5]
        print(f"embedding dataset {dataset_id} with {embedder_id}")
        embeds = [Embedder.get_embeddings(doc) for doc in tqdm(docs)]
        for similarity_id, similarity, similarity_args in SIMILARITIES:
            Similarity = similarity(**similarity_args)
            print(f"calculating similarities with {similarity_id}")
            sims = [Similarity.get_similarities(e) for e in embeds]
            for direction_id, direction, direction_args in DIRECTIONS:
                print(f"updating directions with {direction_id}")
                Direction = direction(**direction_args)
                sims = [Direction.update_directions(s) for s in sims]
                for scorer_id, scorer, scorer_args in SCORERS:
                    Scorer = scorer(**scorer_args)
                    experiment = f"{dataset_id}-{embedder_id}-{similarity_id}-{direction_id}-{scorer_id}"
                    experiment_path = os.path.join(PATH, experiment) 
                    try:
                        if not os.path.exists(experiment_path):
                            os.makedirs(experiment_path)

                        print("running experiment: ", experiment)
                        results = []
                        references = []
                        summaries = []
                        df = pd.DataFrame()

                        for sim, doc in zip(sims, docs):
                            scores = Scorer.get_scores(sim)
                            summary = Summarizer.get_summary(doc, scores)
                            results.append({
                                "num_sects": len(doc.sections),
                                "num_sents": sum([len(s.sentences) for s in doc.sections]),
                                "summary": summary,

                            })
                            summaries.append([s[0] for s in summary])
                            references.append(doc.reference)



                        df['reference'] = references
                        df['system'] = summaries
                        df['meta'] = results 

                        file = os.path.join(experiment_path, 'summaries.csv')
                        out_file = os.path.join(experiment_path, 'scores.csv')
                        df.to_csv(file, encoding='utf-8')

                        if os.path.exists(file):
                            score = Evaluate()
                            score.rouge_scores(file, out_file)

                    except FileExistsError:
                        print(f"{experiment} already exists, skipping...")
                        pass

