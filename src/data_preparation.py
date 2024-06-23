# src/data_preparation.py

import os

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read().splitlines()
    return data

def preprocess_data(data):
    # Check if all words are lowercase
    assert all(word.islower() for word in data), "Not all words are in lowercase."
    return data