from nltk.corpus import wordnet
from util import config
from nltk.corpus import stopwords
import json
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
from tqdm import tqdm
from util.common import get_wordnet_pos
wnl = WordNetLemmatizer()

stop_words = stopwords.words('english')


def emotion_intensity(NRC, word):
    '''
    Function to calculate emotion intensity (Eq. 1 in our paper)
    :param NRC: NRC_VAD vectors
    :param word: query word
    :return:
    '''
    v, a, d = NRC[word]
    a = a/2
    return (np.linalg.norm(np.array([v, a]) - np.array([0.5, 0])) - 0.06467)/0.607468




REMOVE_RELATIONS = ["Antonym", "ExternalURL", "NotDesires", "NotHasProperty", "NotCapableOf", "dbpedia", "DistinctFrom", "EtymologicallyDerivedFrom",
                    "EtymologicallyRelatedTo", "SymbolOf", "FormOf", "AtLocation", "DerivedFrom", "SymbolOf", "CreatedBy", "Synonym", "MadeOf"]


def wordCate(word_pos):
    w_p = get_wordnet_pos(word_pos[1])
    if w_p == wordnet.NOUN or w_p == wordnet.ADV or w_p == wordnet.ADJ or w_p == wordnet.VERB:
        return True
    else:
        return False


def load_concept(dataset, VAD, concept, concept_num, vocab):

    word2index = vocab.word2index
    train_contexts = dataset['context']
    for i, sample in tqdm(enumerate(train_contexts)):
        vads = []  # each item is sentence, each sentence contains a list word' vad vectors
        vad = []
        concepts = []  # concepts of each sample
        total_concepts = []
        total_concepts_tid = []
        for j, sentence in enumerate(sample):
            words_pos = nltk.pos_tag(sentence)

            vads.append([VAD[word] if word in word2index and word in VAD else [
                        0.5, 0.0, 0.5] for word in sentence])
            vad.append([emotion_intensity(VAD, word)
                       if word in VAD else 0.0 for word in sentence])

            sentence_concepts = [
                concept[word] if word in word2index and word not in stop_words and word in concept and wordCate(
                    words_pos[wi]) else []
                for wi, word in enumerate(sentence)]

            sentence_concept_words = []  # for each sentence
            sentence_concept_vads = []
            sentence_concept_vad = []

            # filter concepts of each token, complete their VAD value, select top total_concept_num.
            for cti, uc in enumerate(sentence_concepts):
                concept_words = []  # for each token
                concept_vads = []
                concept_vad = []
                if uc != []:  # this token has concepts
                    # iterate the concept lists [c,r,w] of each token
                    for c in uc:
                        # remove concpets that are stopwords or not in the dict
                        if c[1] not in REMOVE_RELATIONS and c[0] not in stop_words and c[0] in word2index:
                            if c[0] in VAD and emotion_intensity(VAD, c[0]) >= 0.6:
                                concept_words.append(c[0])
                                concept_vads.append(VAD[c[0]])
                                concept_vad.append(
                                    emotion_intensity(VAD, c[0]))
                                # all concepts of a sentence
                                total_concepts.append(c[0])
                                # the token that each concept belongs to
                                total_concepts_tid.append([j, cti])

                    # concept_words = concept_words[:5]
                    # concept_vads = concept_vads[:5]
                    # concept_vad = concept_vad[:5]
                    concept_words = concept_words[:concept_num]
                    concept_vads = concept_vads[:concept_num]
                    concept_vad = concept_vad[:concept_num]

                sentence_concept_words.append(concept_words)
                sentence_concept_vads.append(concept_vads)
                sentence_concept_vad.append(concept_vad)

            sentence_concepts = [sentence_concept_words,
                                 sentence_concept_vads, sentence_concept_vad]
            concepts.append(sentence_concepts)
        dataset['concepts'].append(concepts)
        dataset['sample_concepts'].append([total_concepts, total_concepts_tid])
        dataset['vads'].append(vads)
        dataset['vad'].append(vad)

    train_targets = dataset['target']
    for i, target in enumerate(train_targets):
        # each item is the VAD info list of each target token
        dataset['target_vads'].append([VAD[word] if word in word2index and word in VAD else [
                                      0.5, 0.0, 0.5] for word in target])
        dataset['target_vad'].append([emotion_intensity(
            VAD, word) if word in VAD and word in word2index else 0.0 for word in target])


def aug_kemp(data_tra, data_val, data_tst, vocab, concept_num=3, total_concept_num=10):
    # with open('EmpatheticDialogue/dataset_preproc.json', "r") as f:
    # [data_tra, data_val, data_tst, vocab] = json.load(f)

    VAD = json.load(open(config.data_dir+"/VAD.json", "r", encoding="utf-8"))
    concept = json.load(
        open(config.data_dir+"/ConceptNet_VAD_dict.json", "r", encoding="utf-8"))

    data_tra['concepts'], data_val['concepts'], data_tst['concepts'] = [], [], []
    data_tra['sample_concepts'], data_val['sample_concepts'], data_tst['sample_concepts'] = [], [], []
    data_tra['vads'], data_val['vads'], data_tst['vads'] = [
    ], [], []  # each sentence's vad vectors
    data_tra['vad'], data_val['vad'], data_tst['vad'] = [
    ], [], []  # each word's emotion intensity
    data_tra['target_vad'], data_val['target_vad'], data_tst['target_vad'] = [
    ], [], []  # each target word's emotion intensity
    data_tra['target_vads'], data_val['target_vads'], data_tst['target_vads'] = [
    ], [], []  # each target word's vad vectors
    load_concept(data_tra, VAD, concept, concept_num, vocab)
    load_concept(data_val, VAD, concept, concept_num, vocab)
    load_concept(data_tst, VAD, concept, concept_num, vocab)
    return data_tra, data_val, data_tst, vocab