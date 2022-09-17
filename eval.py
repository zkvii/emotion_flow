import torch
import os
import torch.nn as nn
from torchmetrics.text import cer, wer
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.text.bert import BERTScore
from torchmetrics.functional import bleu_score
from torchmetrics.functional import rouge_score
from nltk.translate.meteor_score import meteor_score
from torchmetrics import BLEUScore, CharErrorRate, CHRFScore, ExtendedEditDistance, MatchErrorRate
from torchmetrics.text.rouge import ROUGEScore
from nltk.corpus import wordnet
from tabulate import tabulate
import nltk
import pandas as pd


import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
def cal_metric(file):
    metric_file=file[:-4]+'_metric.txt'
    metric_file_csv=file[:-4]+'_metric.csv'
    contexts=[]
    emotions = []
    predicts=[]
    beam_predicts=[]
    refs=[]
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('Pred:'):
                predicts.append(line.strip('Pred:').strip())
            elif line.startswith('Ref:'):
                refs.append(line.strip('Ref:').strip())
            elif line.startswith('Emotion:'):
                emotions.append(line.strip('Emotion:').strip())
            elif line.startswith('Beam:'):
                beam_predicts.append(line.strip('Beam:').strip())
            elif line.startswith('Context'):
                contexts.append(line.strip('Context:').strip())
    print(len(emotions),len(predicts),len(beam_predicts))
    # assert len(emotions) == len(predicts) == len(beam_predicts) == len(refs)
    result={
        'bleu1':0,
        'bleu2':0,
        'bleu4':0,
        'beam-bleu1':0,
        'beam-bleu2':0,
        'beam-bleu4':0,
        'bertscore':0,
        'meteor':0,
        'rouge1':0,
        'rouge2':0,
        'rougeL':0,
        'rougeLsum':0,
        'beam-rouge1':0,
        'beam-rouge2':0,
        'beam-rougeL':0,
        'beam-rougeLsum':0,
        'beam-bertscore':0,
        'beam-meteor':0
    }
    ##init metric fun
    # bertscore = BERTScore(model_name_or_path='bert-base-uncased',device='cuda:0')
    # for i in tqdm(range(0,len(predicts),32)):
    #     beam_batch = beam_predicts[i:i+32]
    #     predict_batch = predicts[i:i+32]
    #     ref_batch = refs[i:i+32]
        
    #     # cur_bert_score = bertscore(predict_batch,ref_batch)['f1']
    #     result['bertscore'] += sum(bertscore(predict_batch,ref_batch)['f1'])
    #     result['beam-bertscore'] += sum(bertscore(beam_batch,ref_batch)['f1'])

    # for (context,emotion,ref,pred,beam_pred) in tqdm(zip(contexts,emotions,refs,predicts,beam_predicts)):
    for i in tqdm(range(len(predicts))):
        context = contexts[i]
        emotion = emotions[i]
        ref = refs[i]
        pred = predicts[i]
        if len(beam_predicts)!=0:
            beam_pred = beam_predicts[i]

            result['beam-meteor'] += meteor_score([nltk.word_tokenize(ref)],nltk.word_tokenize(beam_pred))
            
            result['beam-bleu1'] += bleu_score(beam_pred,[ref],n_gram=1)
            result['beam-bleu2'] += bleu_score(beam_pred,[ref],n_gram=2)
            result['beam-bleu4'] += bleu_score(beam_pred,[ref],n_gram=4)
            
            result['beam-rouge1'] += rouge_score(beam_pred,ref,rouge_keys='rouge1')['rouge1_fmeasure']
            result['beam-rouge2'] += rouge_score(beam_pred,ref,rouge_keys='rouge2')['rouge2_fmeasure']
            result['beam-rougeL'] += rouge_score(beam_pred,ref,rouge_keys='rougeL')['rougeL_fmeasure']
            result['beam-rougeLsum'] += rouge_score(beam_pred,ref,rouge_keys='rougeLsum')['rougeLsum_fmeasure']
        # print(context,emotion,ref,pred,beam_pred)
        result['meteor'] += meteor_score([nltk.word_tokenize(ref)],nltk.word_tokenize(pred))
        result['bleu1'] += bleu_score(pred,[ref],n_gram=1)
        result['bleu2'] += bleu_score(pred,[ref],n_gram=2)
        result['bleu4'] += bleu_score(pred,[ref],n_gram=4)
        result['rouge1'] += rouge_score(pred,ref,rouge_keys='rouge1')['rouge1_fmeasure']
        result['rouge2'] += rouge_score(pred,ref,rouge_keys='rouge2')['rouge2_fmeasure']
        result['rougeL'] += rouge_score(pred,ref,rouge_keys='rougeL')['rougeL_fmeasure']
        result['rougeLsum'] += rouge_score(pred,ref,rouge_keys='rougeLsum')['rougeLsum_fmeasure']
    result_list=[(k,float(result[k])/len(refs)) for k in result]
    format_metric=tabulate(result_list,headers=["metric","value"],tablefmt='fancy_grid')
    print(format_metric)
    with open(metric_file,'w') as f:
        f.write(format_metric)
    
    df=pd.DataFrame(result_list,columns=['metric','value'])
    df.to_csv(metric_file_csv,index=False)
    print(format_metric)
if __name__ == '__main__':
    file_path='./predicts'
    #get all files in predicts
    files=os.listdir(file_path) 
    for file in files:
        if file.endswith('results.txt'):
            print(file)
            cal_metric(file_path+'/'+file)