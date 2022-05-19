from datasets import load_metric
import pandas as pd
import torch

class Evaluate(object):
	'''----------------------------------------------------------------
	Initialize evaluation object 
	Args:
		model_path	 	: str
	Return: 
		object
	'''
	def __init__(self):

		self.metric = load_metric("rouge")
		self.device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	'''----------------------------------------------------------------
	Rouge score
	Args:
		pred  : str 
		ref   : str
	Return:
		score : dict
	'''
	def compute_rouge_all(self, pred, ref):

		prediction = []
		prediction.append(pred) 
		reference = []
		reference.append(ref) 
		#compute takes list
		score = self.metric.compute(predictions= prediction, references= reference)

		return score

	'''----------------------------------------------------------------
	Args:
		pred  : str 
		ref   : str 
		r_type: str
	Return:
		r_p, r_r, r_f: float
	'''
	def compute_rouge(self, pred, ref, r_type):

		prediction = []
		prediction.append(pred) 
		reference = []
		reference.append(ref) 
		#compute takes list
		score = self.metric.compute(predictions= prediction, references= reference, rouge_types=[r_type])[r_type].mid
		r_p = round(score.precision, 4)
		r_r = round(score.recall, 4)
		r_f = round(score.fmeasure, 4)

		return r_p, r_r, r_f

	'''----------------------------------------------------------------
	Args:
		file: file
	Return: None
	'''
	def rouge_scores(self, file, out_file):

		df = pd.read_csv(file)
		sdf = pd.DataFrame()
		sdf['all_scores'] = df.apply(lambda x: self.compute_rouge_all(x['system'], x['reference']), axis=1)
		
		sdf['r1_precision'], sdf['r1_recall'], sdf['r1_fscore'] \
		= zip(*df.apply(lambda x: self.compute_rouge(x['system'], x['reference'], "rouge1"), axis=1))
		
		sdf['r2_precision'], sdf['r2_recall'], sdf['r2_fscore'] \
		= zip(*df.apply(lambda x: self.compute_rouge(x['system'], x['reference'], "rouge2"), axis=1))

		sdf['rL_precision'], sdf['rL_recall'], sdf['rL_fscore'] \
		= zip(*df.apply(lambda x: self.compute_rouge(x['system'], x['reference'], "rougeL"), axis=1))

		
		sdf.to_csv(out_file, encoding='utf-8')