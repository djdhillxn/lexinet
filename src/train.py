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
        self.counter = defaultdict(int)
        self.unigrams = Counter()
        self.unigrams_rev = Counter()
        self.continuation_counts = defaultdict(Counter)
        self.continuation_counts_rev = defaultdict(Counter)
        self.smoothing_factor = 0.1
        self.D = 0.75
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"

    def save_all(self, models_dir):
        data = {
            'ngrams': self.ngrams,
            'ngrams_rev': self.ngrams_rev,
            'continuation_counts': self.continuation_counts,
            'continuation_counts_rev': self.continuation_counts_rev,
            'unigrams': self.unigrams,
            'unigrams_rev': self.unigrams_rev
        }

        path = os.path.join(models_dir, f"n_{self.n}_gram_model_kneser_ney.pkl")
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def train_gold(self, data):
        for word in tqdm(data):
            padded_word = ['<s>'] * (self.n - 1) + list(word) + ['</s>'] * (self.n - 1)
            for gram in ngrams(padded_word, self.n):
                prefix = gram[:-1]
                suffix = gram[-1]
                if suffix == '</s>':
                    continue
                self.ngrams[prefix][suffix] += 1
                if self.n > 2:
                    potential_mask_indices = [0]*(self.n-1)
                    cnt = 0
                    for i in range(self.n - 1):
                        if gram[i] not in ['<s>', '</s>']:
                            potential_mask_indices[i] = 1
                            cnt += 1
                        
                    num_masks = max(cnt-1, self.n - 2)
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

    def train(self, data):
        for word in tqdm(data):
            padded_word = ['<s>'] * (self.n - 1) + list(word) + ['</s>'] * (self.n - 1)
            for gram in ngrams(padded_word, self.n):
                prefix = gram[:-1]
                suffix = gram[-1]
                if suffix == '</s>':
                    continue
                self.ngrams[prefix][suffix] += 1
                self.unigrams[suffix] += 1
                self.continuation_counts[suffix][prefix] += 1
                if self.n > 2:
                    potential_mask_indices = [0] * (self.n - 1)
                    cnt = 0
                    for i in range(self.n - 1):
                        if gram[i] not in ['<s>', '</s>']:
                            potential_mask_indices[i] = 1
                            cnt += 1
                    num_masks = max(cnt - 1, self.n - 2)
                    if num_masks > 0:
                        for i in range(1, num_masks + 1):
                            for mask_indices in combinations([j for j, x in enumerate(potential_mask_indices) if x == 1], i):
                                masked_gram = list(gram[:-1])
                                for index in mask_indices:
                                    masked_gram[index] = '_'
                                masked_gram = tuple(masked_gram)
                                prefix = masked_gram[:]
                                suffix = gram[-1]
                                self.ngrams[prefix][suffix] += 1
                                self.continuation_counts[suffix][prefix] += 1

    def train_reverse_gold(self, data):
        for word in tqdm(data):
            padded_word = ['<s>'] * (self.n - 1) + list(word) + ['</s>'] * (self.n - 1)
            for gram in ngrams(padded_word, self.n):
                prefix = gram[0]
                suffix = gram[1:]
                if prefix == '<s>':
                    continue
                self.ngrams_rev[suffix][prefix] += 1
                if self.n > 2:
                    potential_mask_indices = [0] * (self.n - 1)
                    cnt = 0
                    for i in range(self.n-1):
                        if gram[i+1] not in ['<s>', '</s>']:
                            potential_mask_indices[i] = 1
                            cnt += 1

                    num_masks = max(cnt-1, self.n - 2)
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

    def train_reverse(self, data):
        for word in tqdm(data):
            padded_word = ['<s>'] * (self.n - 1) + list(word) + ['</s>'] * (self.n - 1)
            for gram in ngrams(padded_word, self.n):
                prefix = gram[0]
                suffix = gram[1:]
                if prefix == '<s>':
                    continue
                self.ngrams_rev[suffix][prefix] += 1
                self.unigrams_rev[prefix] += 1
                self.continuation_counts_rev[prefix][suffix] += 1
                if self.n > 2:
                    potential_mask_indices = [0] * (self.n - 1)
                    cnt = 0
                    for i in range(self.n - 1):
                        if gram[i + 1] not in ['<s>', '</s>']:
                            potential_mask_indices[i] = 1
                            cnt += 1
                    num_masks = max(cnt - 1, self.n - 2)
                    if num_masks > 0:
                        for i in range(1, num_masks + 1):
                            for mask_indices in combinations([j for j, x in enumerate(potential_mask_indices) if x == 1], i):
                                masked_gram = list(gram[1:])
                                for index in mask_indices:
                                    masked_gram[index] = '_'
                                masked_gram = tuple(masked_gram)
                                prefix = gram[0]
                                suffix = masked_gram[:]
                                self.ngrams_rev[suffix][prefix] += 1
                                self.continuation_counts_rev[prefix][suffix] += 1

    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self.ngrams, file)

    def save_rev(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self.ngrams_rev, file)

# Train the n-gram model
n = 3
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

ngram_model.save_all(model_dir)


"""
# Save the n-gram model
model_path = os.path.join(model_dir, f"n_{n}_gram_model_new_age.pkl")
ngram_model.save(model_path)

# Save the n-gram model
model_path_rev = os.path.join(model_dir, f"n_{n}_gram_model_rev_new_age.pkl")
ngram_model.save_rev(model_path_rev)
"""