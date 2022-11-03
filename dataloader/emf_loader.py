from collections import defaultdict
import os
from typing import Dict, List
import nltk
import json
import torch
import pickle
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from util.constants import WORD_PAIRS as word_pairs
from util.constants import EMO_MAP as emo_map
from util.constants import MAP_EMO as map_emo
from util.constants import EMO_MAP_T as emo_map_t
from util.constants import MAP_EMO_T as map_emo_t
# from util.constants import EMO_MAP_ORIGIN as emo_map_o
# from util.constants import EMO_MAP_RANDOM as emo_map_r
from util.constants import DATA_FILES, SPECIAL_TOKEN_INDEX
from util import config
from util.common import get_wordnet_pos
import os
import numpy as np
import logging
from nltk.corpus import wordnet

emotion_lexicon = json.load(open("data/NRCDict.json"))[0]
stop_words = stopwords.words("english")


class EMFLang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.word2count = {}
        self.wordEmoScore = {}
        self.n_words = 0
        if config.emotion_emb_type == "order":
            self.emo2index = emo_map
            self.index2emo = map_emo
        elif config.emotion_emb_type == "tolerance":
            self.emo2index = emo_map_t
            self.index2emo = map_emo_t
        # self.temo2index = emo_map_t
        # self.index2temo = map_emo_t
        self.set_special_token()

    def set_special_token(self):
        for token in SPECIAL_TOKEN_INDEX:
            self.index_word(token)

    def init_word_emo_score(self):
        for word in self.word2index:
            self.wordEmoScore[word] = 0.0

    def get_word_emo_score(self, word):
        return self.wordEmoScore[word]

    def add_words_emo_score(self, words, score):
        for word in words:
            self.add_word_emo_score(word, score)

    def add_word_emo_score(self, word, score):
        self.wordEmoScore[word] += score

    def index_words(self, sentence: List[str]):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word: str):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def process_sent(sentence):
    """replace unofficial exp with official exp and 
    """
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)
    sentence = nltk.word_tokenize(sentence)
    return sentence


def encode_context(vocab, items, data_dict):
    for ctx in tqdm(items):
        ctx_list = []
        emo_list = []
        for i, c in enumerate(ctx):
            # item is sentence word list
            item = process_sent(c)
            ctx_list.append(item)
            # make vocab
            vocab.index_words(item)
            # tag word in sentence
            ws_pos = nltk.pos_tag(item)  # pos
            for w in ws_pos:
                w_p = get_wordnet_pos(w[1])
                if w[0] not in stop_words and (
                    w_p == wordnet.ADJ or w[0] in emotion_lexicon
                ):
                    # word that is not a stop word and marked as adjective or in emotion_lexicon will be added to e_list
                    emo_list.append(w[0])

        data_dict["context"].append(ctx_list)
        data_dict["lexical_context"].append(emo_list)

    return data_dict


def encode(vocab: EMFLang, data: List):

    data_dict = {
        "context": [],
        "target": [],
        "emotion": [],
        "situation": [],
        "lexical_context": []
    }

    if config.code_check:
        context_list = data[0][:20]
        target_list = data[1][:20]
        emotion_list = data[2][:20]
        situation_list = data[3][:20]
    else:
        context_list=data[0]
        target_list = data[1]
        emotion_list = data[2]
        situation_list = data[3]
    #process context
    encode_context(vocab, context_list, data_dict)
    #process target
    data_dict['emotion'] = emotion_list
    for item in tqdm(target_list):
        item = process_sent(item)
        data_dict['target'].append(item)
        vocab.index_words(item)

    for item in tqdm(situation_list):
        item = process_sent(item)
        data_dict['situation'].append(item)
        vocab.index_words(item)
    # ignore situation
    assert (
        len(data_dict["context"])
        == len(data_dict["target"])
        == len(data_dict["emotion"])
        == len(data_dict["situation"])
        == len(data_dict["lexical_context"])
    )

    return data_dict


def read_files(vocab):
    """load encoded dataset
    """
    files = DATA_FILES(config.data_dir)
    # list[dialog,target,emotion,situation] 4 elements array
    train_files = [np.load(f, allow_pickle=True) for f in files["train"]]
    dev_files = [np.load(f, allow_pickle=True) for f in files["dev"]]
    test_files = [np.load(f, allow_pickle=True) for f in files["test"]]
    data_train = encode(vocab, train_files)
    data_dev = encode(vocab, dev_files)
    data_test = encode(vocab, test_files)
    # data augmentation
    return data_train, data_dev, data_test, vocab


def flatten(t):
    """flatten a list of list
    """
    return [item for sublist in t for item in sublist]


def load_dataset():
    """load preprocessed dataset if exits
    """
    data_dir = config.data_dir
    # cache_file = f"{data_dir}/dataset_preproc.p"
    cache_file = f"{data_dir}/ds_{config.model}_{config.emotion_emb_type}.p"

    vocab = EMFLang()

    if os.path.exists(cache_file):
        print("LOADING empathetic_dialogue")
        with open(cache_file, "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    else:
        print("Building dataset...")

        # resort emotion
        emos = [k for k in vocab.emo2index]
        vocab.index_words(emos)
        # not just readfiles but with some modification
        data_tra, data_val, data_tst, vocab = read_files(
            vocab
        )
        with open(cache_file, "wb") as f:
            pickle.dump([data_tra, data_val, data_tst, vocab], f)
            print("Saved PICKLE")

    for i in range(3):
        print("[situation]:", " ".join(data_tra["situation"][i]))
        print("[emotion]:", data_tra["emotion"][i])
        print("[context]:", [" ".join(u) for u in data_tra["context"][i]])
        print("[target]:", " ".join(data_tra["target"][i]))
        print('-------------------------\n')
    return data_tra, data_val, data_tst, vocab

def fill_list_with_maximum(l):
    """fill list with maximum value
    """
    maximum = sorted(l)[-1]
    fill_val=maximum*2+1
    return [fill_val if num == -1 else num for num in l]

class EMFDataset(Dataset):

    def __init__(self, data, vocab:EMFLang):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data
        self.analyzer = SentimentIntensityAnalyzer()

    def __len__(self):
        return len(self.data["target"])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        # 5 basic elements
        item["context_text"] = self.data["context"][index]
        item["situation_text"] = self.data["situation"][index]
        item["target_text"] = self.data["target"][index]
        item["emotion_text"] = [self.data["emotion"][index]]
        item["lexical_context_text"] = self.data["lexical_context"][index]
        # context emo score for mime
        context_text = " ".join(flatten(self.data["context"][index]))

        context_emos = self.analyzer.polarity_scores(
            context_text
        )
        item["context_emotion_scores"] = [context_emos['neg'],
                                          context_emos['neu'],
                                          context_emos['pos']]
        # get emotion casue words and related words
        # get emotion mask
        # word2vec

        return self.process(item)

    def process(self, item):
        data_sequence = {}
        for key in item:
            if key == "emotion_text" or key == "situation_text" or key=="target_text":
                sequence = [config.SOS_idx]+[
                    self.vocab.word2index[word]
                    if word in self.vocab.word2index else
                    config.UNK_idx for word in item[key]] \
                    + [config.EOS_idx]
                sequence = torch.LongTensor(sequence)
            if key == "context_text":
                sequence = []
                #to divide the context into 2 parts
                mask_sequence=[]
                word_count_sequence=[]
                for i,ctx in enumerate(item[key]):
                    one_turn_sequence=[config.SOS_idx]
                    one_turn_count=[-1]
                    for word in ctx:
                        if word in self.vocab.word2index:
                            one_turn_sequence.append(self.vocab.word2index[word])
                            if word in item['lexical_context_text']:
                                one_turn_count.append(self.vocab.word2index[word])
                            else:
                                one_turn_count.append(-1)
                                # one_turn_key_mask.append(0)
                        else:
                            one_turn_sequence.append(config.UNK_idx)
                            #unknown word focused
                            one_turn_count.append(47)
                            
                    one_turn_sequence.append(config.EOS_idx)
                    one_turn_count.append(self.vocab.word2count['<EOS>'])
                    # sequence.extend([config.SOS_idx]+[
                    #     self.vocab.word2index[word]
                    #     if word in self.vocab.word2index else
                    #     config.UNK_idx for word in ctx]
                    #     + [config.EOS_idx])
                    sequence.extend(one_turn_sequence)
                    word_count_sequence.extend(one_turn_count)
                    spk = (
                        self.vocab.word2index["<USR>"]
                        if i % 2 == 0
                        else self.vocab.word2index["<SYS>"]
                    )

                    mask_sequence.extend([spk for _ in range(len(ctx)+2)])
                mask_sequence = torch.LongTensor(mask_sequence)
                word_count_sequence=fill_list_with_maximum(word_count_sequence)
                word_count_seq=torch.LongTensor(word_count_sequence)
                data_sequence['input_divide_mask'] = mask_sequence
                data_sequence['word_count_sequence'] = word_count_seq

                
                sequence = torch.LongTensor(sequence)
            if key == "context_emotion_scores":
                sequence = torch.FloatTensor(item[key])
            # if key == 'lexical_context_text':
            #     sequence = []
            #     for i,ctx in enumerate(item[key]):
            #         sequence.extend([config.SOS_idx]+[
            #             self.vocab.word2index[word]
            #             if word in self.vocab.word2index else
            #             config.UNK_idx for word in ctx]
            #             + [config.EOS_idx])
            #     sequence = torch.LongTensor(sequence)
            data_sequence[key+'_tensor'] = sequence
            data_sequence[key] = item[key]
        return data_sequence


def pad_sequence(data_seq,is_context=False):
    max_len = max([t.shape[0] for t in data_seq])
    if not is_context:
        for i in range(len(data_seq)):
            data_seq[i] = torch.cat([data_seq[i], torch.zeros(
                max_len-data_seq[i].shape[0]).fill_(config.PAD_idx)])
    else:
        #add a cls token
        max_len+=1
        for i in range(len(data_seq)):
            data_seq[i] = torch.cat([torch.LongTensor([config.CLS_idx]),data_seq[i],torch.zeros(
                max_len-data_seq[i].shape[0]).fill_(config.PAD_idx)])
        
    return torch.stack(data_seq)

def collate_fn(data):
    # only tokenize wihout mask
    context = []
    target = []
    situation = []
    emotion = []
    context_emo_score = []
    input_divide_mask = []
    context_text=[]
    target_text=[]
    emotion_text=[]
    lexical=[]
    lexical_weight=[]
    for item in data:
        context.append(item['context_text_tensor'])
        target.append(item['target_text_tensor'])
        emotion.append(item['emotion_text_tensor'])
        context_emo_score.append(item['context_emotion_scores_tensor'])
        situation.append(item['situation_text_tensor'])
        input_divide_mask.append(item['input_divide_mask'])

        context_text.append(item['context_text'])
        target_text.append(item['target_text'])
        emotion_text.append(item['emotion_text'])
    context = pad_sequence(context,is_context=True).long()
    target = pad_sequence(target).long()
    emotion = pad_sequence(emotion).long()
    context_emo_score = pad_sequence(context_emo_score).float()
    input_divide_mask = pad_sequence(input_divide_mask,is_context=True).long()
    # situation = pad_sequence(situation)

    return {
        'input_batch':context,
        'target_batch':target,
        'emotion_batch':emotion,
        'context_emo_batch':context_emo_score,
        'input_divide_mask':input_divide_mask,
        'context_text':context_text,
        'target_text':target_text,
        'emotion_text':emotion_text
        # 'situation_batch':situation
    }


def prepare_data_seq(batch_size=32):

    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()
    # word2index, word2count, index2word, n_words = vocab
    logging.info("Vocab  {} ".format(vocab.n_words))

    num_workers = 1
    dataset_train = EMFDataset(pairs_tra, vocab)
    data_loader_tra = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=True
    )

    dataset_valid = EMFDataset(pairs_val, vocab)
    data_loader_val = DataLoader(
        dataset=dataset_valid,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    dataset_test = EMFDataset(pairs_tst, vocab)
    data_loader_tst = DataLoader(
        dataset=dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn,
        num_workers=num_workers
    )
    return (
        data_loader_tra,
        data_loader_val,
        data_loader_tst,
        vocab
    )
