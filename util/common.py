import json
import os
import torch
import numpy as np
from util import config
import yaml
from torchmetrics.text.bert import BERTScore
from torchmetrics import BLEUScore, CharErrorRate, CHRFScore, ExtendedEditDistance, MatchErrorRate
from torchmetrics.text.rouge import ROUGEScore
from nltk.corpus import wordnet


def set_seed():
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.seed)


def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def save_config():
    if not config.test:
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
        with open(config.save_path + "/config.txt", "w") as the_file:
            for k, v in config.args.__dict__.items():
                if "False" in str(v):
                    continue
                elif "True" in str(v):
                    the_file.write("--{} ".format(k))
                else:
                    the_file.write("--{} {} ".format(k, v))


def embedding_similarity(
    batch,
    similarity: str = "cosine",
    reduction: str = "none",
    zero_diagonal: bool = True,
):
    """
    Computes representation similarity
    Example:
        >>> from torchmetrics.functional import embedding_similarity
        >>> embeddings = torch.tensor([[1., 2., 3., 4.], [1., 2., 3., 4.], [4., 5., 6., 7.]])
        >>> embedding_similarity(embeddings)
        tensor([[0.0000, 1.0000, 0.9759],
                [1.0000, 0.0000, 0.9759],
                [0.9759, 0.9759, 0.0000]])
    Args:
        batch: (batch, dim)
        similarity: 'dot' or 'cosine'
        reduction: 'none', 'sum', 'mean' (all along dim -1)
        zero_diagonal: if True, the diagonals are set to zero
    Return:
        A square matrix (batch, batch) with the similarity scores between all elements
        If sum or mean are used, then returns (b, 1) with the reduced value for each row
    """
    if similarity == "cosine":
        norm = torch.norm(batch, p=2, dim=1)
        batch = batch / norm.unsqueeze(1)

    sqr_mtx = batch.mm(batch.transpose(1, 0))

    if zero_diagonal:
        sqr_mtx = sqr_mtx.fill_diagonal_(0)

    if reduction == "mean":
        sqr_mtx = sqr_mtx.mean(dim=-1)

    if reduction == "sum":
        sqr_mtx = sqr_mtx.sum(dim=-1)

    return sqr_mtx


def print_opts(opts):
    """Prints the values of all command-line arguments."""
    print("=" * 80)
    print("Opts".center(80))
    print("-" * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print("{:>30}: {:<30}".format(key, opts.__dict__[key]).center(80))
    print("=" * 80)


def save_best_path(file_path):

    file_name = f'./best_model/{config.model}-{config.emotion_emb_type}.json'
    model_map = {'path': file_path}
    with open(file_name, 'w+') as f:
        json.dump(model_map, f)


def load_best_path():
    file_name = f'./best_model/{config.model}-{config.emotion_emb_type}.json'
    with open(file_name) as f:
        model_map = json.load(f)
    return model_map['path']


def save_best_hparams(model):
    logger = model.logger
    file_path = f'{logger.log_dir}/{logger.NAME_HPARAMS_FILE}'
    with open(file_path, 'w') as f:
        yaml.dump(vars(config.args), f, default_flow_style=False)


def cal_metric(file):
    context=[]
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
    print(len(emotions),len(predicts),len(beam_predicts))
    assert len(emotions) == len(predicts) == len(beam_predicts) == len(refs)
    result={
        'bleu1':0,
        'bleu2':0,
        'bleu4':0,
        'beam-bleu1':0,
        'beam-bleu2':0,
        'beam-bleu4':0,
        'bertscore':0,
        'rouge1':0,
        'rouge2':0,
        'rougeL':0,
        'rougeLsum':0,
        'beam-rouge1':0,
        'beam-rouge2':0,
        'beam-rougeL':0,
        'beam-rougeLsum':0,
        'beam-bertscore':0,
    }
    ##init metric fun
    bertscore = BERTScore(device='cuda').to('cuda')

    for (emotion,predict,bpredict,ref) in zip(emotions,predicts,beam_predicts,refs):
        f1 = bertscore(predict,ref)['f1']
        # bert_score = bertscore(predict,ref)
    print(result['bertscore'])