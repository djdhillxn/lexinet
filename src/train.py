# src/train.py

import nltk
from nltk.util import ngrams
from collections import defaultdict, Counter
import pickle
import os
from data_preparation import load_data, preprocess_data
from tqdm import tqdm
from itertools import combinations

class NgramModel:
    def __init__(self, n):
        self.n = n
        self.ngrams = defaultdict(Counter)
        self.ngrams_rev = defaultdict(Counter)
    
    def train_old(self, data):
        for word in tqdm(data):
            padded_word = ['<s>'] * (self.n - 1) + list(word) + ['</s>']
            word_length = len(word)
            for gram in ngrams(padded_word, self.n):
                prefix = gram[:-1]
                suffix = gram[-1]
                if suffix == '</s>':
                    continue
                self.ngrams[prefix][suffix] += 1
                if self.n > 2:
                    if gram[-2] != '<s>':
                        masked_gram = list(gram)
                        masked_gram[-2] = '_'
                        masked_gram = tuple(masked_gram)
                        prefix = masked_gram[:-1]
                        suffix = masked_gram[-1]
                        self.ngrams[prefix][suffix] += 1
                    if gram[-3] != '<s>':
                        masked_gram = list(gram)
                        masked_gram[-3] = '_'
                        masked_gram = tuple(masked_gram)
                        prefix = masked_gram[:-1]
                        suffix = masked_gram[-1]
                        self.ngrams[prefix][suffix] += 1

    def train_reverse_old(self, data):
        for word in tqdm(data):
            padded_word = ['<s>'] * (self.n - 1) + list(word) + ['</s>']
            word_length = len(word)
            for gram in ngrams(padded_word, self.n):
                prefix = gram[0]
                suffix = gram[1:]
                if prefix == '<s>':
                    continue
                self.ngrams_rev[suffix][prefix] += 1
                if self.n > 2:
                    if gram[-1] != '</s>':
                        masked_gram = list(gram)
                        masked_gram[-1] = '_'
                        masked_gram = tuple(masked_gram)
                        prefix = masked_gram[0]
                        suffix = masked_gram[1:]
                        self.ngrams_rev[suffix][prefix] += 1
                    if gram[1] != '<s>':
                        masked_gram = list(gram)
                        masked_gram[1] = '_'
                        masked_gram = tuple(masked_gram)
                        prefix = masked_gram[0]
                        suffix = masked_gram[1:]
                        self.ngrams_rev[suffix][prefix] += 1

    def train(self, data):
        for word in tqdm(data):
            padded_word = ['<s>'] * (self.n - 1) + list(word) + ['</s>']
            for gram in ngrams(padded_word, self.n):
                prefix = gram[:-1]
                suffix = gram[-1]
                if suffix == '</s>':
                    continue
                self.ngrams[prefix][suffix] += 1
                if self.n > 2:
                    potential_mask_indices = [0]*(self.n-1)
                    cnt = 0
                    for i in range(self.n-1):
                        if gram[i] not in ['<s>', '</s>']:
                            potential_mask_indices[i] = 1
                            cnt += 1
                        
                    num_masks = max(cnt-1, self.n-2)
                    if num_masks > 0:
                        for i in range(1, num_masks+1):
                            for mask_indices in combinations([j for j, x in enumerate(potential_mask_indices) if x == 1], i):
                                masked_gram = list(gram[:-1])
                                for index in mask_indices:
                                    masked_gram[index] = '_'
                                masked_gram = tuple(masked_gram)
                                prefix = masked_gram[:]
                                suffix = gram[-1]
                                self.ngrams[prefix][suffix] += 1

    def train_reverse(self, data):
        for word in tqdm(data):
            padded_word = ['<s>'] * (self.n - 1) + list(word) + ['</s>']
            for gram in ngrams(padded_word, self.n):
                prefix = gram[0]
                suffix = gram[1:]
                if prefix == '<s>':
                    continue
                self.ngrams_rev[suffix][prefix] += 1
                if self.n > 2:
                    potential_mask_indices = [0]*(self.n-1)
                    cnt = 0
                    for i in range(self.n-1):
                        if gram[i+1] not in ['<s>', '</s>']:
                            potential_mask_indices[i] = 1
                            cnt += 1

                    num_masks = max(cnt-1, self.n-2)
                    if num_masks > 0:
                        for i in range(1, num_masks+1):
                            for mask_indices in combinations([j for j, x in enumerate(potential_mask_indices) if x == 1], i):
                                masked_gram = list(gram[1:])
                                for index in mask_indices:
                                    masked_gram[index] = '_'
                                masked_gram = tuple(masked_gram)
                                prefix = gram[0]
                                suffix = masked_gram[:]
                                self.ngrams_rev[suffix][prefix] += 1


    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self.ngrams, file)

    def save_rev(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self.ngrams_rev, file)

# Train the n-gram model
n = 2
ngram_model = NgramModel(n)

train_data_path = "data/train/words_train.txt"
test_data_path = "data/test/words_test.txt"

# Load and preprocess train and test data
train_data = preprocess_data(load_data(train_data_path))
test_data = preprocess_data(load_data(test_data_path))

ngram_model.train(train_data)
ngram_model.train_reverse(train_data)

model_dir = "results/models"
os.makedirs(model_dir, exist_ok=True)

# Save the n-gram model
model_path = os.path.join(model_dir, f"n_{n}_gram_model_n.pkl")
ngram_model.save(model_path)

# Save the n-gram model
model_path_rev = os.path.join(model_dir, f"n_{n}_gram_model_rev_n.pkl")
ngram_model.save_rev(model_path_rev)