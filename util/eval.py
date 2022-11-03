import collections
from email import header
import math
import ast
from operator import index
import pdb
import os
import re
import pandas as pd
from tabulate import tabulate

def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned by this
        methods.
    Returns:
        The Counter containing all n-grams upto max_order in segment
        with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def _compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    """Computes BLEU score of translated segments against one or more references.
    Args:
        reference_corpus: list of lists of references for each translation. Each
            reference should be tokenized into a list of tokens.
        translation_corpus: list of translations to score. Each translation
            should be tokenized into a list of tokens.
        max_order: Maximum n-gram order to use when computing BLEU score.
        smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
        3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
            precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    
    for (references, translation) in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


def get_dist(res):
    unigrams = []
    bigrams = []
    avg_len = 0.
    ma_dist1, ma_dist2 = 0., 0.
    for q, r in res.items():
        ugs = r
        bgs = []
        i = 0
        while i < len(ugs) - 1:
            bgs.append(ugs[i] + ugs[i + 1])
            i += 1
        unigrams += ugs
        bigrams += bgs
        ma_dist1 += len(set(ugs)) / (float)(len(ugs) + 1e-16)
        ma_dist2 += len(set(bgs)) / (float)(len(bgs) + 1e-16)
        avg_len += len(ugs)
    n = len(res)
    ma_dist1 /= n
    ma_dist2 /= n
    mi_dist1 = len(set(unigrams)) / (float)(len(unigrams))
    mi_dist2 = len(set(bigrams)) / (float)(len(bigrams))
    avg_len /= n
    return ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len



def cal_one_model(file, opt=''):
    if opt!='':
        opt_f = open(opt, 'w')
    else:
        opt_f = open(file.rstrip('.txt')+'_metric.txt', 'w')

    # print(file)
    # opt_f.write(file +'\n')
    result={'experiment':file.rstrip('.txt')}
    # result={}
    with open(file, 'r') as f:
        target = []
        pred = []
        beam_preds=[]

        res = {}
        beam_res={}
        itr = 0
        beam_iter=0
        for line in f.readlines():
            if line.startswith('Pred:'):
                p = line.strip('Pred:').strip()
                pls = p.split()
                pred.append(pls)
                res[itr] = pls
                itr += 1
                # p = re.sub(r'([a-zA-Z])([,;?.!:\'/])', r"\1 \2", p)
            if line.startswith('Ref:'):
                t = line.strip('Ref:').strip()
                tls = t.split()
                target.append([tls])
            if line.startswith('Beam:'):
                b = line.strip('Beam:').strip()
                bls = b.split()
                beam_preds.append(bls)
                beam_res[beam_iter]=bls
                beam_iter+=1

        bleu1 = _compute_bleu(target, pred, max_order=1)
        bleu2 = _compute_bleu(target, pred, max_order=2)
        bleu4 = _compute_bleu(target, pred, max_order=4)

        bbleu1=_compute_bleu(target, beam_preds, max_order=1)
        bbleu2=_compute_bleu(target, beam_preds, max_order=2)
        bbleu4=_compute_bleu(target, beam_preds, max_order=4)
        result.update( {
            "bleu1": round(bleu1[0]*100,2),
            "bleu2": round(bleu2[0]*100,2),
            "bleu4": round(bleu4[0]*100,2),
            "bbleu1": round(bbleu1[0]*100,2),
            "bbleu2": round(bbleu2[0]*100,2),
            "bbleu4": round(bbleu4[0]*100,2)
        })

    ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len = get_dist(res)
    
    beam_ma_dist1, beam_ma_dist2, beam_mi_dist1, beam_mi_dist2, beam_avg_len = get_dist(beam_res)
    result.update({
        "ma_dist1":ma_dist1,
        "ma_dist2":ma_dist2,
        "mi_dist1":mi_dist1,
        "mi_dist2":mi_dist2,
        "avg_len":avg_len,
        "beam_ma_dist1":beam_ma_dist1,
        "beam_ma_dist2":beam_ma_dist2,
        "beam_mi_dist1":beam_mi_dist1,
        "beam_mi_dist2":beam_mi_dist2,
        "beam_avg_len":beam_avg_len,
    })
    result_list=[(k,result[k]) for k in result]
    format_metric=tabulate(result_list,headers=["metric","value"],tablefmt='fancy_grid')
    print(format_metric)
    opt_f.write(format_metric)

    df=pd.DataFrame(result,index=[0])
    df['Dist-1']=mi_dist1*100
    df['Dist-2']=mi_dist2*100
    df.to_csv(file.rstrip('.txt')+'_metric.csv',index=False)



if __name__ == '__main__':
    cal_one_model('./predicts/emf-tolerance-results.txt')

