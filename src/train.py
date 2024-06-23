# src/train.py

import nltk
from nltk.util import ngrams
from collections import defaultdict, Counter
import pickle
import os
from data_preparation import load_data, preprocess_data
from tqdm import tqdm

class NgramModel:
    def __init__(self, n):
        self.n = n
        self.ngrams = defaultdict(Counter)

    def train(self, data):
        for word in tqdm(data):
            padded_word = ['<s>'] * (self.n - 1) + list(word) + ['</s>']
            for gram in ngrams(padded_word, self.n):
                prefix = gram[:-1]
                suffix = gram[-1]
                self.ngrams[prefix][suffix] += 1

    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self.ngrams, file)

# Train the n-gram model
n = 2
ngram_model = NgramModel(n)

train_data_path = "../data/train/words_train.txt"
test_data_path = "../data/test/words_test.txt"

# Load and preprocess train and test data
train_data = preprocess_data(load_data(train_data_path))
test_data = preprocess_data(load_data(test_data_path))

ngram_model.train(train_data)

model_dir = "../results/models"
os.makedirs(model_dir, exist_ok=True)

# Save the n-gram model
model_path = os.path.join(model_dir, f"n_{n}_gram_model_corrected.pkl")
ngram_model.save(model_path)