import unicodedata
import re
import os
from tqdm import tqdm
import numpy as np
import json
# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOC_token = 1  # Start-of-sentence token
EOC_token = 2  # End-of-sentence token

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {"pad": PAD_token, "soc": SOC_token, "eoc": EOC_token}
        self.word2count = {"pad": 0, "soc": 0, "eoc": 0}
        self.index2word = {PAD_token: "pad", SOC_token: "soc", EOC_token: "eoc"}
        self.num_words = 3  # Count SOC, EOC, PAD

    def addCaption(self, caption):
        
        for word in caption.split(' '):
            self.addWord(word)

    def addWord(self, word):
        
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {"pad": PAD_token, "soc": SOC_token, "eoc": EOC_token}
        self.word2count = {"pad": 0, "soc": 0, "eoc": 0}
        self.index2word = {PAD_token: "pad", SOC_token: "soc", EOC_token: "eoc"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)

    def save_vocabulary(self):
        save_json_file(self.word2index, "word2index.json")
        save_json_file(self.index2word, "index2word.json")
        

    def load_vocabulary(self):
        self.word2index = load_json_file("word2index.json")
        self.index2word = load_json_file("index2word.json")
        


def save_json_file(dict, path):
    

    with open(path, "w") as file:
        json.dump(dict, file)


def load_json_file(path):
    with open(path, "r") as f:
        file = json.load(f)
    return file
    


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeCaption(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def normalizeAllCaptions(all_captions):
    normalized_captions = []
    print("Normalizing captions...")
    for caption in tqdm(all_captions):   
        normalized_captions.append(normalizeCaption(caption))

    return normalized_captions
