from torch.utils.data import DataLoader, Dataset
from util import config
from dataloader.loader import Lang, process_sent
import os
import pickle
from util.constants import EMO_MAP_ORIGIN, WORD_PAIRS as word_pairs
from util.constants import EMO_MAP as emo_map
from util.constants import EMO_MAP_T as emo_map_t
from util.constants import EMO_MAP_ORIGIN as emo_map_o
from util.constants import EMO_MAP_RANDOM as emo_map_r
from util.constants import DATA_FILES
import numpy as np
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas


def encode_emf(vocab: Lang, files):
    """encode the dataset
    """
    data = {"context": [], "target": [], "emotion": [],
        "situation": [], "emotion_context": []}
    for file in files:
        for dialog, target, emotion, situation in zip(file[0], file[1], file[2], file[3]):
            # dialog: list of list of str
            # target: list of str
            # emotion: str
            # situation: list of str
            # emotion_context: list of str
            # context: list of list of str
            # emotion_context = [emo_map[emotion]] + situation
            emotion_context = [emo_map[emotion]] + situation
            context = []
            for i, utterance in enumerate(dialog):
                if i % 2 == 0:
                    context.append([config.USR_idx] + utterance)
                else:
                    context.append([config.SYS_idx] + utterance)
            data["context"].append(context)
            data["target"].append(target)
            data["emotion"].append(emo_map[emotion])
            data["situation"].append(situation
    for file in files:
        for i in range(len(file[0])):
            # context
            context=[]
            for j in range(len(file[0][i])):
                context.append(process_sent(file[0][i][j], vocab))
            # target
            target=process_sent(file[1][i], vocab)
            # emotion
            emotion=emo_map[file[2][i]]
            # situation
            situation=process_sent(file[3][i], vocab)
            data["context"].append(context)
            data["target"].append(target)
            data["emotion"].append(emotion)
            data["situation"].append(situation)
    return data

def read_emf_files(vocab: Lang):
    """read the dataset files
    """

    files=DATA_FILES(config.data_dir)
    # list[dialog,target,emotion,situation] 4 elements array
    train_files=[np.load(f, allow_pickle=True) for f in files["train"]]
    dev_files=[np.load(f, allow_pickle=True) for f in files["dev"]]
    test_files=[np.load(f, allow_pickle=True) for f in files["test"]]
    data_train=encode_emf(vocab, train_files)
    data_dev=encode_emf(vocab, dev_files)
    data_test=encode_emf(vocab, test_files)
    return data_tra, data_val, data_tst, vocab



def load_emf_dataset():
    """load preprocessed dataset if exits
    """
    data_dir=config.data_dir
    # cache_file = f"{data_dir}/dataset_preproc.p"
    cache_file=f"{data_dir}/emotion_{config.emotion_emb_type}.p"

    if os.path.exists(cache_file):
        print("LOADING PROCESSED EMPATHETIC DIALOGUE DATASET")
        with open(cache_file, "rb") as f:
            [data_tra, data_val, data_tst, vocab]=pickle.load(f)
    else:
        print("Building dataset...")

        vocab=Lang(
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
        emos=[k for k in emo_map.keys()]
        vocab.index_words(emos)
        # not just readfiles but with some modification
        data_tra, data_val, data_tst, vocab=read_files(
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
        print("[emotion_context]:,", " ".join(data_tra["emotion_context"][i]))
        print('-------------------------')
    return data_tra, data_val, data_tst, vocab
