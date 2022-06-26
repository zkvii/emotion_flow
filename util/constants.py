DATA_FILES = lambda data_dir: {
    "train": [
        f"{data_dir}/sys_dialog_texts.train.npy",
        f"{data_dir}/sys_target_texts.train.npy",
        f"{data_dir}/sys_emotion_texts.train.npy",
        f"{data_dir}/sys_situation_texts.train.npy",
    ],
    "dev": [
        f"{data_dir}/sys_dialog_texts.dev.npy",
        f"{data_dir}/sys_target_texts.dev.npy",
        f"{data_dir}/sys_emotion_texts.dev.npy",
        f"{data_dir}/sys_situation_texts.dev.npy",
    ],
    "test": [
        f"{data_dir}/sys_dialog_texts.test.npy",
        f"{data_dir}/sys_target_texts.test.npy",
        f"{data_dir}/sys_emotion_texts.test.npy",
        f"{data_dir}/sys_situation_texts.test.npy",
    ],
}

WORD_PAIRS = {
    "it's": "it is",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "you'd": "you would",
    "you're": "you are",
    "you'll": "you will",
    "i'm": "i am",
    "they're": "they are",
    "that's": "that is",
    "what's": "what is",
    "couldn't": "could not",
    "i've": "i have",
    "we've": "we have",
    "can't": "cannot",
    "i'd": "i would",
    "i'd": "i would",
    "aren't": "are not",
    "isn't": "is not",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "there's": "there is",
    "there're": "there are",
}

EMO_MAP_ORIGIN = {
    "surprised": 0,
    "excited": 1,
    "annoyed": 2,
    "proud": 3,
    "angry": 4,
    "sad": 5,
    "grateful": 6,
    "lonely": 7,
    "impressed": 8,
    "afraid": 9,
    "disgusted": 10,
    "confident": 11,
    "terrified": 12,
    "hopeful": 13,
    "anxious": 14,
    "disappointed": 15,
    "joyful": 16,
    "prepared": 17,
    "guilty": 18,
    "furious": 19,
    "nostalgic": 20,
    "jealous": 21,
    "anticipating": 22,
    "embarrassed": 23,
    "content": 24,
    "devastated": 25,
    "sentimental": 26,
    "caring": 27,
    "trusting": 28,
    "ashamed": 29,
    "apprehensive": 30,
    "faithful": 31,
}

MAP_EMO_ORIGIN = {
    0: "surprised",
    1: "excited",
    2: "annoyed",
    3: "proud",
    4: "angry",
    5: "sad",
    6: "grateful",
    7: "lonely",
    8: "impressed",
    9: "afraid",
    10: "disgusted",
    11: "confident",
    12: "terrified",
    13: "hopeful",
    14: "anxious",
    15: "disappointed",
    16: "joyful",
    17: "prepared",
    18: "guilty",
    19: "furious",
    20: "nostalgic",
    21: "jealous",
    22: "anticipating",
    23: "embarrassed",
    24: "content",
    25: "devastated",
    26: "sentimental",
    27: "caring",
    28: "trusting",
    29: "ashamed",
    30: "apprehensive",
    31: "faithful",
}

EMO_MAP_T={

    'angry':0,
    'annoyed':0,
    'jealous':0,
    'furious':0,

    'afraid':1,
    'terrified':1,
    'anxious':1,
    'apprehensive':1,

    'sad':2,
    'disappointed':2,
    'devastated':2,
    'lonely':2,
    'nostalgic':2,
    'sentimental':2,
      
    'disgusted':3,
    'embarrassed':3,
    'guilty':3,
    'ashamed':3,
    
    'faithful':4,
    'trusting':4,
    'grateful':4,
    'caring':4,
    'hopeful':4,
    
    'anticipating':5,
    'prepared':5,
    'confident':5,
        
    'proud':6,
    'impressed':6,
    'content':6,
    
    'excited':7,
    'surprised':7,
    'joyful':7,
    
}

MAP_EMO_T={
    0:'angry',
    1:'afraid',
    2:'sad',
    3:'disgusted',
    4:'faithful',
    5:'anticipating',
    6:'proud',
    7:'exited',
}

EMO_MAP={
    'angry':0,
    'annoyed':1,
    'jealous':2,
    'furious':3,

    'afraid':4,
    'terrified':5,
    'anxious':6,
    'apprehensive':7,

    'sad':8,
    'disappointed':9,
    'devastated':10,
    'lonely':11,
    'nostalgic':12,
    'sentimental':13,
      
    'disgusted':14,
    'embarrassed':15,
    'guilty':16,
    'ashamed':17,
    
    'faithful':18,
    'trusting':19,
    'grateful':20,
    'caring':21,
    'hopeful':22,
    
    'anticipating':23,
    'prepared':24,
    'confident':25,
        
    'proud':26,
    'impressed':27,
    'content':28,
    
    'excited':29,
    'surprised':30,
    'joyful':31,
}

MAP_EMO={EMO_MAP[k]:k for k in EMO_MAP}
EMO_MAP_RANDOM={}
MAP_EMO_RANDOM={}