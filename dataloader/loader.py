from collections import defaultdict
import os
import nltk
import json
import torch
import pickle
from tqdm.auto import tqdm
import torch.utils.data as data
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dataloader.concept_preprocess import aug_kemp
from util.constants import EMO_MAP_ORIGIN, WORD_PAIRS as word_pairs
from util.constants import EMO_MAP as emo_map
from util.constants import EMO_MAP_T as emo_map_t
from util.constants import EMO_MAP_ORIGIN as emo_map_o
from util.constants import EMO_MAP_RANDOM as emo_map_r
from util.constants import DATA_FILES
from util import config
from util.common import get_wordnet_pos
import os
import numpy as np
import logging
from nltk.corpus import wordnet

relations = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"]
emotion_lexicon = json.load(open("data/NRCDict.json"))[0]
stop_words = stopwords.words("english")


class Lang:
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def process_sent(sentence):
    """replace unofficial exp with official exp and 
    tokenize sentence

    Parameters
    ----------
    sentence : str
        _description_

    Returns
    -------
    sentence:str
        _description_
    """
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)
    sentence = nltk.word_tokenize(sentence)
    return sentence


def get_commonsense(comet, items):
    # item:List[32List[]]
    # get common sense by different relations
    cs_list = []
    input_events = [" ".join(item) for item in items]
    # cs_old=[]
    # for rel in relations:
    #     cs_res = comet.generate(input_event, rel)
    #     cs_res = [process_sent(item) for item in cs_res]
    #     cs_old.append(cs_res)
    cs_list = comet.generate_f(input_events, relations)
    # cs_list=
    # data_dict["utt_cs"].append(cs_list)
    return cs_list

# to use batch accelerate gen commonsense corpus


def encode_ctx(vocab, items, data_dict, comet):
    # ctx is turns of dialogue
    print(f'start context encoding')
    print(f'using large size i.e. 32')
    for ctx in tqdm(items):
        ctx_list = []
        e_list = []
        commonsense_list = []
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
                    e_list.append(w[0])
            # only gen last one
            if i == len(ctx) - 1:
                # commonsense_list = get_commonsense(comet, item, data_dict)
                commonsense_list = get_commonsense(comet, item)
        # raw context list
        #commonsense_list : List[5*RelationList[n*GenElement]]
        data_dict["utt_cs"].append(commonsense_list)
        data_dict["context"].append(ctx_list)
        # emotion word list list
        data_dict["emotion_context"].append(e_list)


def encode_context(vocab, items, data_dict, comet):
    # items = items[:1024]
    commonsense_item = []
    for ctx in tqdm(items):
        ctx_list = []
        e_list = []
        # commonsense_list=[]
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
                    e_list.append(w[0])
            # only gen last one
            if i == len(ctx) - 1:
                commonsense_item.append(item)
                # commonsense_list = get_commonsense(comet, item, data_dict)
                # commonsense_list = get_commonsense(comet, item)
        # raw context list
        #commonsense_list : List[5*RelationList[n*GenElement]]
        # data_dict["utt_cs"].append(commonsense_list)
        data_dict["context"].append(ctx_list)
        # emotion word list list
        data_dict["emotion_context"].append(e_list)
    # batch_items=np.reshape(commonsense_item,[])

    for context in tqdm(range(0, len(commonsense_item), 32)):
        chunks = commonsense_item[context:context+32]
        chunks_gen = get_commonsense(comet, chunks)
        data_dict["utt_cs"].extend(chunks_gen)
    return data_dict


def encode(vocab, files):
    """encode files(list[str]) with vocab by comet model

    Parameters
    ----------
    vocab : _type_
        _description_
    files : _type_
        _description_

    Returns
    -------
    data_dict
        six key indicates different item
    """
    from model.comet import Comet

    data_dict = {
        "context": [],
        "target": [],
        "emotion": [],
        "situation": [],
        "emotion_context": [],
        "utt_cs": [],
    }
    comet = Comet("./data/ED/comet")

    for i, k in enumerate(data_dict.keys()):
        items = files[i][:100]
        # items = files[i]
        if k == "context":
            # encoding context

            # using comet model to gen commonsense data
            # encode_ctx(vocab, items, data_dict, comet)
            encode_context(vocab, items, data_dict, comet)
        elif k == "emotion":
            data_dict[k] = items
        else:
            # sentence field process
            # just replace unofficial expression
            for item in tqdm(items):
                item = process_sent(item)
                data_dict[k].append(item)
                vocab.index_words(item)
        if i == 3:
            break
    assert (
        len(data_dict["context"])
        == len(data_dict["target"])
        == len(data_dict["emotion"])
        == len(data_dict["situation"])
        == len(data_dict["emotion_context"])
        == len(data_dict["utt_cs"])
    )

    return data_dict


def read_files(vocab):
    """load encoded dataset

    Parameters
    ----------
    vocab : _type_
        encoded vocabulary

    Returns
    -------
    tuple(list,list,list)
    """
    files = DATA_FILES(config.data_dir)
    # list[dialog,target,emotion,situation] 4 elements array
    train_files = [np.load(f, allow_pickle=True) for f in files["train"]]
    dev_files = [np.load(f, allow_pickle=True) for f in files["dev"]]
    test_files = [np.load(f, allow_pickle=True) for f in files["test"]]
    # data augmentation
    data_train = encode(vocab, train_files)
    data_dev = encode(vocab, dev_files)
    data_test = encode(vocab, test_files)
    # return raw data
    # kemp augmentation
    # get_concept_dict(vocab)
    # rank_concept_dict()
    aug_kemp(data_train, data_dev, data_test, vocab)
    #
    return data_train, data_dev, data_test, vocab


def flatten(t):
    return [item for sublist in t for item in sublist]


def load_dataset():
    """load preprocessed dataset if exits

    Returns
    -------
    _type_
        _description_
    """
    data_dir = config.data_dir
    # cache_file = f"{data_dir}/dataset_preproc.p"
    cache_file = f"{data_dir}/ds_{config.emotion_emb_type}.p"

    if os.path.exists(cache_file):
        print("LOADING empathetic_dialogue")
        with open(cache_file, "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    else:
        print("Building dataset...")

        vocab = Lang(
            {
                config.UNK_idx: "<UNK>",
                config.PAD_idx: "<PAD>",
                config.EOS_idx: "<EOS>",
                config.SOS_idx: "<SOS>",
                config.USR_idx: "<USR>",
                config.SYS_idx: "<SYS>",
                config.KG_idx: "<KG>",
                config.CLS_idx: "<CLS>",
                config.SEP_idx: "<SEP>",
            }
        )
        emos=[k for k in EMO_MAP_ORIGIN]
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
        print('[concept of context]:')
        for si, sc in enumerate(data_tra['concepts'][i]):
            print('concept of sentence {} : {}'.format(si, flatten(sc[0])))
        print("[target]:", " ".join(data_tra["target"][i]))
        print("[emotion_context]:,", " ".join(data_tra["emotion_context"][i]))
        print(" ")
    return data_tra, data_val, data_tst, vocab


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, vocab):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data
        if config.emotion_emb_type == 'order':
            self.emo_map = emo_map
        elif config.emotion_emb_type == 'tolerance':
            self.emo_map = emo_map_t
        elif config.emotion_emb_type == 'origin':
            self.emo_map = emo_map_o
        else:
            self.emo_map = emo_map_r
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
        item["emotion_text"] = self.data["emotion"][index]
        item["emotion_context"] = self.data["emotion_context"][index]
        # context emo score for mime
        item["context_emotion_scores"] = self.analyzer.polarity_scores(
            " ".join(self.data["context"][index][0])
        )

        item["context"], item["context_mask"] = self.preprocess(
            item["context_text"])
        item["target"] = self.preprocess(item["target_text"], anw=True)
        item["emotion"], item["emotion_label"] = self.preprocess_emo(
            item["emotion_text"], self.emo_map
        )
        (
            item["emotion_context"],
            item["emotion_context_mask"],
        ) = self.preprocess(item["emotion_context"])

        # cem data
        item["cs_text"] = self.data["utt_cs"][index]
        item["x_intent_txt"] = item["cs_text"][0]
        item["x_need_txt"] = item["cs_text"][1]
        item["x_want_txt"] = item["cs_text"][2]
        item["x_effect_txt"] = item["cs_text"][3]
        item["x_react_txt"] = item["cs_text"][4]

        item["x_intent"] = self.preprocess(item["x_intent_txt"], cs=True)
        item["x_need"] = self.preprocess(item["x_need_txt"], cs=True)
        item["x_want"] = self.preprocess(item["x_want_txt"], cs=True)
        item["x_effect"] = self.preprocess(item["x_effect_txt"], cs=True)
        item["x_react"] = self.preprocess(item["x_react_txt"], cs="react")

        # kemp data
        inputs = self.preprocess([self.data["context"][index],
                                  self.data["vads"][index],
                                  self.data["vad"][index],
                                  self.data["concepts"][index]], kemp=True)
        item["kemp_context"], item["context_ext"], item["context_mask"], item["vads"], item["vad"], \
            item["concept_text"], item["concept"], item["concept_ext"], item["concept_vads"], item["concept_vad"], \
            item["oovs"] = inputs
        # which is the same as prior
        # item["target_kemp"] = self.preprocess(item["target_text"], anw=True)
        item["target_ext"] = self.target_oovs(
            item["target_text"], item["oovs"])
        item["emotion"], item["emotion_label"] = self.preprocess_emo(item["emotion_text"],
                                                                     self.emo_map)  # one-hot and scalor label
        item["emotion_widx"] = self.vocab.word2index[item["emotion_text"]]
        return item

    def target_oovs(self, target, oovs):
        ids = []
        for w in target:
            if w not in self.vocab.word2index:
                if w in oovs:
                    ids.append(len(self.vocab.word2index) + oovs.index(w))
                else:
                    ids.append(config.UNK_idx)
            else:
                ids.append(self.vocab.word2index[w])
        ids.append(config.EOS_idx)
        return torch.LongTensor(ids)

    def process_oov(self, context, concept):  #
        ids = []
        oovs = []
        for si, sentence in enumerate(context):
            for w in sentence:
                if w in self.vocab.word2index:
                    i = self.vocab.word2index[w]
                    ids.append(i)
                else:
                    if w not in oovs:
                        oovs.append(w)
                    oov_num = oovs.index(w)
                    ids.append(len(self.vocab.word2index) + oov_num)

        for sentence_concept in concept:
            for token_concept in sentence_concept:
                for c in token_concept:
                    if c not in oovs and c not in self.vocab.word2index:
                        oovs.append(c)
        return ids, oovs

    def preprocess(self, arr, anw=False, cs=None, emo=False, kemp=False):
        """Converts words to ids."""
        # normal convert
        if anw:
            sequence = [
                self.vocab.word2index[word]
                if word in self.vocab.word2index
                else config.UNK_idx
                for word in arr
            ] + [config.EOS_idx]

            return torch.LongTensor(sequence)
        # convert commensense field
        elif cs:
            sequence = [config.CLS_idx] if cs != "react" else []
            for sent in arr:
                sequence += [
                    self.vocab.word2index[word]
                    for word in sent
                    if word in self.vocab.word2index and word not in ["to", "none"]
                ]

            return torch.LongTensor(sequence)
        # convert emotion field
        elif emo:
            x_emo = [config.CLS_idx]
            x_emo_mask = [config.CLS_idx]
            for i, ew in enumerate(arr):
                x_emo += [
                    self.vocab.word2index[ew]
                    if ew in self.vocab.word2index
                    else config.UNK_idx
                ]
                x_emo_mask += [self.vocab.word2index["CLS"]]

            assert len(x_emo) == len(x_emo_mask)
            return torch.LongTensor(x_emo), torch.LongTensor(x_emo_mask)
        # convert kemp field
        elif kemp:
            context = arr[0]
            context_vads = arr[1]
            context_vad = arr[2]
            concept = [arr[3][l][0] for l in range(len(arr[3]))]
            concept_vads = [arr[3][l][1] for l in range(len(arr[3]))]
            concept_vad = [arr[3][l][2] for l in range(len(arr[3]))]

            X_dial = [config.CLS_idx]
            X_dial_ext = [config.CLS_idx]
            X_mask = [config.CLS_idx]  # for dialogue state
            X_vads = [[0.5, 0.0, 0.5]]
            X_vad = [0.0]

            X_concept_text = defaultdict(list)
            X_concept = [[]]  # 初始值是cls token
            X_concept_ext = [[]]
            X_concept_vads = [[0.5, 0.0, 0.5]]
            X_concept_vad = [0.0]
            assert len(context) == len(concept)

            X_ext, X_oovs = self.process_oov(context, concept)
            X_dial_ext += X_ext

            for i, sentence in enumerate(context):
                X_dial += [self.vocab.word2index[word]
                           if word in self.vocab.word2index else config.UNK_idx for word in sentence]
                spk = self.vocab.word2index["<USR>"] if i % 2 == 0 else self.vocab.word2index["<SYS>"]
                X_mask += [spk for _ in range(len(sentence))]
                X_vads += context_vads[i]
                X_vad += context_vad[i]

                for j, token_conlist in enumerate(concept[i]):
                    if token_conlist == []:
                        X_concept.append([])
                        X_concept_ext.append([])
                        X_concept_vads.append([0.5, 0.0, 0.5])  # ??
                        X_concept_vad.append(0.0)
                    else:
                        X_concept_text[sentence[j]
                                       ] += token_conlist[:config.concept_num]
                        X_concept.append(
                            [self.vocab.word2index[con_word] if con_word in self.vocab.word2index else config.UNK_idx for con_word in token_conlist[:config.concept_num]])

                        con_ext = []
                        for con_word in token_conlist[:config.concept_num]:
                            if con_word in self.vocab.word2index:
                                con_ext.append(self.vocab.word2index[con_word])
                            else:
                                if con_word in X_oovs:
                                    con_ext.append(X_oovs.index(
                                        con_word) + len(self.vocab.word2index))
                                else:
                                    con_ext.append(config.UNK_idx)
                        X_concept_ext.append(con_ext)
                        X_concept_vads.append(
                            concept_vads[i][j][:config.concept_num])
                        X_concept_vad.append(
                            concept_vad[i][j][:config.concept_num])

                        assert len([self.vocab.word2index[con_word] if con_word in self.vocab.word2index else config.UNK_idx for con_word in token_conlist[:config.concept_num]]) == len(
                            concept_vads[i][j][:config.concept_num]) == len(concept_vad[i][j][:config.concept_num])
            assert len(X_dial) == len(X_mask) == len(
                X_concept) == len(X_concept_vad) == len(X_concept_vads)

            return X_dial, X_dial_ext, X_mask, X_vads, X_vad, \
                X_concept_text, X_concept, X_concept_ext, X_concept_vads, X_concept_vad, \
                X_oovs
        # undefined
        else:
            x_dial = [config.CLS_idx]
            x_mask = [config.CLS_idx]
            for i, sentence in enumerate(arr):
                x_dial += [
                    self.vocab.word2index[word]
                    if word in self.vocab.word2index
                    else config.UNK_idx
                    for word in sentence
                ]
                spk = (
                    self.vocab.word2index["<USR>"]
                    if i % 2 == 0
                    else self.vocab.word2index["<SYS>"]
                )
                x_mask += [spk for _ in range(len(sentence))]
            assert len(x_dial) == len(x_mask)

            return torch.LongTensor(x_dial), torch.LongTensor(x_mask)

    def preprocess_emo(self, emotion, emo_map):
        '''
            one-hot encode
        '''
        program = [0] * len(emo_map)
        program[emo_map[emotion]] = 1
        return program, emo_map[emotion]


def collate_fn(data):
    def merge(sequences):
        """pad sequences with 1 to max len

        Parameters
        ----------
        sequences : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(
            len(sequences), max(lengths)
        ).long()  # padding to max_sentence with 1
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        return padded_seqs, lengths

    def merge_concept(samples, samples_ext, samples_vads, samples_vad):
        concept_lengths = []  # 每个sample的concepts数目
        token_concept_lengths = []  # 每个sample的每个token的concepts数目
        concepts_list = []
        concepts_ext_list = []
        concepts_vads_list = []
        concepts_vad_list = []

        for i, sample in enumerate(samples):
            length = 0  # 记录当前样本总共有多少个concept，
            sample_concepts = []
            sample_concepts_ext = []
            token_length = []
            vads = []
            vad = []

            for c, token in enumerate(sample):
                if token == []:  # 这个token没有concept
                    token_length.append(0)
                    continue
                length += len(token)
                token_length.append(len(token))
                sample_concepts += token
                sample_concepts_ext += samples_ext[i][c]
                vads += samples_vads[i][c]
                vad += samples_vad[i][c]

            if length > config.total_concept_num:
                value, rank = torch.topk(torch.LongTensor(
                    vad), k=config.total_concept_num)

                new_length = 1
                new_sample_concepts = [config.SEP_idx]  # for each sample
                new_sample_concepts_ext = [config.SEP_idx]
                new_token_length = []
                new_vads = [[0.5, 0.0, 0.5]]
                new_vad = [0.0]

                cur_idx = 0
                for ti, token in enumerate(sample):
                    if token == []:
                        new_token_length.append(0)
                        continue
                    top_length = 0
                    for ci, con in enumerate(token):
                        point_idx = cur_idx + ci
                        if point_idx in rank:
                            top_length += 1
                            new_length += 1
                            new_sample_concepts.append(con)
                            new_sample_concepts_ext.append(
                                samples_ext[i][ti][ci])
                            new_vads.append(samples_vads[i][ti][ci])
                            new_vad.append(samples_vad[i][ti][ci])
                            assert len(samples_vads[i][ti][ci]) == 3

                    new_token_length.append(top_length)
                    cur_idx += len(token)

                new_length += 1  # for sep token
                new_sample_concepts = [config.SEP_idx] + new_sample_concepts
                new_sample_concepts_ext = [
                    config.SEP_idx] + new_sample_concepts_ext
                new_vads = [[0.5, 0.0, 0.5]] + new_vads
                new_vad = [0.0] + new_vad

                # the number of concepts including SEP
                concept_lengths.append(new_length)
                # the number of tokens which have concepts
                token_concept_lengths.append(new_token_length)
                concepts_list.append(new_sample_concepts)
                concepts_ext_list.append(new_sample_concepts_ext)
                concepts_vads_list.append(new_vads)
                concepts_vad_list.append(new_vad)
                assert len(new_sample_concepts) == len(new_vads) == len(new_vad) == len(
                    new_sample_concepts_ext), "The number of concept tokens, vads [*,*,*], and vad * should be the same."
                assert len(new_token_length) == len(token_length)
            else:
                length += 1
                sample_concepts = [config.SEP_idx] + sample_concepts
                sample_concepts_ext = [config.SEP_idx] + sample_concepts_ext
                vads = [[0.5, 0.0, 0.5]] + vads
                vad = [0.0] + vad

                concept_lengths.append(length)
                token_concept_lengths.append(token_length)
                concepts_list.append(sample_concepts)
                concepts_ext_list.append(sample_concepts_ext)
                concepts_vads_list.append(vads)
                concepts_vad_list.append(vad)

        if max(concept_lengths) != 0:
            # padding index 1 (bsz, max_concept_len); add 1 for root
            padded_concepts = torch.ones(
                len(samples), max(concept_lengths)).long()
            # padding index 1 (bsz, max_concept_len)
            padded_concepts_ext = torch.ones(
                len(samples), max(concept_lengths)).long()
            padded_concepts_vads = torch.FloatTensor([[[0.5, 0.0, 0.5]]]).repeat(
                len(samples), max(concept_lengths), 1)  # padding index 1 (bsz, max_concept_len)
            padded_concepts_vad = torch.FloatTensor([[0.0]]).repeat(
                len(samples), max(concept_lengths))  # padding index 1 (bsz, max_concept_len)
            padded_mask = torch.ones(len(samples), max(
                concept_lengths)).long()  # concept(dialogue) state

            for j, concepts in enumerate(concepts_list):
                end = concept_lengths[j]
                if end == 0:
                    continue
                padded_concepts[j, :end] = torch.LongTensor(concepts[:end])
                padded_concepts_ext[j, :end] = torch.LongTensor(
                    concepts_ext_list[j][:end])
                padded_concepts_vads[j, :end, :] = torch.FloatTensor(
                    concepts_vads_list[j][:end])
                padded_concepts_vad[j, :end] = torch.FloatTensor(
                    concepts_vad_list[j][:end])
                padded_mask[j, :end] = config.KG_idx  # for DIALOGUE STATE

            return padded_concepts, padded_concepts_ext, concept_lengths, padded_mask, token_concept_lengths, padded_concepts_vads, padded_concepts_vad
        else:  # there is no concept in this mini-batch
            return torch.Tensor([]), torch.LongTensor([]), torch.LongTensor([]), torch.BoolTensor([]), torch.LongTensor([]), torch.Tensor([]), torch.Tensor([])

    def merge_vad(vads_sequences, vad_sequences):  # for context
        lengths = [len(seq) for seq in vad_sequences]
        padding_vads = torch.FloatTensor([[[0.5, 0.0, 0.5]]]).repeat(
            len(vads_sequences), max(lengths), 1)
        padding_vad = torch.FloatTensor([[0.5]]).repeat(
            len(vads_sequences), max(lengths))

        for i, vads in enumerate(vads_sequences):
            end = lengths[i]  # the length of context
            padding_vads[i, :end, :] = torch.FloatTensor(vads[:end])
            padding_vad[i, :end] = torch.FloatTensor(vad_sequences[i][:end])
        # (bsz, max_context_len, 3); (bsz, max_context_len)
        return padding_vads, padding_vad

    def adj_mask(context, context_lengths, concepts, token_concept_lengths):
        '''

        :param self:
        :param context: (bsz, max_context_len)
        :param context_lengths: [] len=bsz
        :param concepts: (bsz, max_concept_len)
        :param token_concept_lengths: [] len=bsz;
        :return:
        '''
        bsz, max_context_len = context.size()
        max_concept_len = concepts.size(1)  # include sep token
        adjacency_size = max_context_len + max_concept_len
        # todo padding index 1, 1=True
        adjacency = torch.ones(bsz, max_context_len, adjacency_size)

        for i in range(bsz):
            # ROOT -> TOKEN
            adjacency[i, 0, :context_lengths[i]] = 0
            adjacency[i, :context_lengths[i], 0] = 0

            con_idx = max_context_len+1       # add 1 because of sep token
            for j in range(context_lengths[i]):
                adjacency[i, j, j - 1] = 0  # TOEKN_j -> TOKEN_j-1

                token_concepts_length = token_concept_lengths[i][j]
                if token_concepts_length == 0:
                    continue
                else:
                    adjacency[i, j, con_idx:con_idx+token_concepts_length] = 0
                    adjacency[i, 0, con_idx:con_idx+token_concepts_length] = 0
                    con_idx += token_concepts_length
        return adjacency

    data.sort(key=lambda x: len(x["context"]),
              reverse=True)  # sort by source seq
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]
    # assert len(item_info['context']) == len(item_info['vad'])
    # input
    input_batch, input_lengths = merge(item_info["context"])
    context_ext_batch, _ = merge(item_info['context_ext'])
    mask_input, mask_input_lengths = merge(item_info["context_mask"])
    emotion_batch, emotion_lengths = merge(item_info["emotion_context"])
    # dialogue context vad
    # (bsz, max_context_len, 3); (bsz, max_context_len)
    context_vads_batch, context_vad_batch = merge_vad(
        item_info['vads'], item_info['vad'])

    # assert input_batch.size(1) == context_vad_batch.size(1)
    ## concepts, vads, vad
    concept_inputs = merge_concept(item_info['concept'],
                                   item_info['concept_ext'],
                                   item_info["concept_vads"],
                                   item_info["concept_vad"])  # (bsz, max_concept_len)
    concept_batch, concept_ext_batch, concept_lengths, mask_concept, token_concept_lengths, concepts_vads_batch, concepts_vad_batch = concept_inputs
    ## adja_mask (bsz, max_context_len, max_context_len+max_concept_len)
    if concept_batch.size()[0] != 0:
        adjacency_mask_batch = adj_mask(
            input_batch, input_lengths, concept_batch, token_concept_lengths)
    else:
        adjacency_mask_batch = torch.Tensor([])
    # Target
    target_batch, target_lengths = merge(item_info["target"])
    target_ext_batch, _ = merge(item_info['target_ext'])
    # input_batch = input_batch
    # mask_input = mask_input
    # target_batch = target_batch

    d = {}
    d["input_batch"] = input_batch
    d["input_lengths"] = torch.LongTensor(input_lengths)
    d["mask_input"] = mask_input
    d["target_batch"] = target_batch
    d["target_lengths"] = torch.LongTensor(target_lengths)
    d["emotion_context_batch"] = emotion_batch
    d["context_ext_batch"] = context_ext_batch  # (bsz, max_context_len)
    # program
    d["emotion_widx"] = torch.LongTensor(item_info['emotion_widx'])
    d["target_program"] = item_info["emotion"]
    d["program_label"] = item_info["emotion_label"]

    # text
    d["input_txt"] = item_info["context_text"]
    d["target_txt"] = item_info["target_text"]
    d["program_txt"] = item_info["emotion_text"]
    d["situation_txt"] = item_info["situation_text"]
    d["context_vads"] = context_vads_batch  # (bsz, max_context_len, 3)
    d["context_vad"] = context_vad_batch  # (bsz, max_context_len)
    d["context_emotion_scores"] = item_info["context_emotion_scores"]
    d["concept_txt"] = item_info['concept_text']
    d["oovs"] = item_info["oovs"]
    # concept
    d["concept_batch"] = concept_batch  # (bsz, max_concept_len)
    d["concept_ext_batch"] = concept_ext_batch  # (bsz, max_concept_len)
    d["concept_lengths"] = torch.LongTensor(concept_lengths)  # (bsz)
    d["mask_concept"] = mask_concept  # (bsz, max_concept_len)
    d["concept_vads_batch"] = concepts_vads_batch  # (bsz, max_concept_len, 3)
    d["concept_vad_batch"] = concepts_vad_batch   # (bsz, max_concept_len)
    d["adjacency_mask_batch"] = adjacency_mask_batch.bool()

    # assert d["emotion_widx"].size() == d["program_label"].size()
    relations = ["x_intent", "x_need", "x_want", "x_effect", "x_react"]
    for r in relations:
        pad_batch, _ = merge(item_info[r])
        pad_batch = pad_batch
        d[r] = pad_batch
        d[f"{r}_txt"] = item_info[f"{r}_txt"]
    # for k in d:
        # if type(d[k]) is torch.Tensor:
        # d[k] = d[k].detach()

    return d


def prepare_data_seq(batch_size=32):

    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()
    # word2index, word2count, index2word, n_words = vocab
    logging.info("Vocab  {} ".format(vocab.n_words))

    num_workers = 1
    dataset_train = Dataset(pairs_tra, vocab)
    data_loader_tra = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=True
    )

    dataset_valid = Dataset(pairs_val, vocab)
    data_loader_val = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    dataset_test = Dataset(pairs_tst, vocab)
    data_loader_tst = torch.utils.data.DataLoader(
        dataset=dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn,
        num_workers=num_workers
    )
    # save_config()
    return (
        data_loader_tra,
        data_loader_val,
        data_loader_tst,
        vocab,
        len(dataset_train.emo_map),
    )


if __name__ == '__main__':
    train_loader, val_loader, test_loader, vocab, emotion_len = prepare_data_seq(
        16)
    sample_batch = next(iter(train_loader))
    print('hello')
