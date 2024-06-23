# src/evaluate.py

import nltk
from nltk.util import ngrams
from collections import defaultdict, Counter
import pickle
import os
from data_preparation import load_data, preprocess_data
from tqdm import tqdm
import math

class NgramModel:
    def __init__(self, n):
        self.n = n
        self.ngrams = None

    def load(self, file_path):
        with open(file_path, 'rb') as file:
            self.ngrams = pickle.load(file)

    def perplexity(self, data):
        total_log_prob = 0
        total_length = 0
        floor_value = 1e-6  # Updated floor value

        for word in data:
            padded_word = ['<s>'] * (self.n - 1) + list(word) + ['</s>']
            for gram in ngrams(padded_word, self.n):
                prefix = gram[:-1]
                suffix = gram[-1]
                count_prefix = sum(self.ngrams[prefix].values())
                count_suffix = self.ngrams[prefix][suffix]
                probability = count_suffix / count_prefix if count_prefix > 0 else floor_value

                if probability <= 0:
                    #print(f"Debug: Non-positive probability encountered. prefix={prefix}, suffix={suffix}, count_prefix={count_prefix}, count_suffix={count_suffix}")
                    probability = floor_value

                total_log_prob += -math.log2(probability)
                total_length += 1

        return math.pow(2, total_log_prob / total_length)

# Load and preprocess test data
test_data_path = "../data/test/words_test.txt"
test_data = preprocess_data(load_data(test_data_path))
model_dir = "../results/models"

# Evaluate n-gram models for n=2 to 7
for n in range(2, 8):
    model_path = os.path.join(model_dir, f"n_{n}_gram_model_corrected.pkl")
    ngram_model = NgramModel(n)
    ngram_model.load(model_path)
    perplexity = ngram_model.perplexity(test_data)
    print(f"Perplexity for {n}-gram model: {perplexity}")