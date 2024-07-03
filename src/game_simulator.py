from data_preparation import load_data, preprocess_data
from player_agent import GreedyPlayer
from collections import defaultdict
import pickle
import os
from tqdm import tqdm

class GameSimulator:
    def __init__(self, word_list_path, models_dir, max_lives=6, num_games=1):
        self.word_list_path = word_list_path
        self.max_lives = max_lives
        self.num_games = num_games
        self.word_list = preprocess_data(load_data(word_list_path))
        self.ngram_models, self.ngram_models_rev = self.load_all_models(models_dir)
        self.word_length_to_n = self.get_word_length_to_n()
        self.player = GreedyPlayer(self.word_length_to_n, self.ngram_models, self.ngram_models_rev)
        
    def get_word_length_to_n(self):
        word_length_to_n = {}
        for l in range(1, 50):
            if l in range(6, 50):
                word_length_to_n[l] = 6
            elif l in range(4, 6):
                word_length_to_n[l] = 5
            elif l in [3]:
                word_length_to_n[l] = 2
            else:
                word_length_to_n[l] = 2
        return word_length_to_n
    
    def create_word_length_to_n(self, n):
        word_length_to_n = {}
        for l in range(1, 50):
            word_length_to_n[l] = n
        return word_length_to_n

    def load_all_models(self, models_dir):
        n_values = [2, 3, 4, 5, 6]
        ngram_models = {}
        ngram_models_rev = {}
        for n in n_values:
            model_name = f"n_{n}_gram_model_new_age.pkl"
            rev_model_name = f"n_{n}_gram_model_rev_new_age.pkl"
            model_path = os.path.join(models_dir, model_name)
            rev_ngram_model_path = os.path.join(models_dir, rev_model_name)
            with open(model_path, 'rb') as file:
                ngram_models[n] = pickle.load(file)
                print("loading", model_path)
            with open(rev_ngram_model_path, 'rb') as file:
                ngram_models_rev[n] = pickle.load(file)
                print("loading", rev_ngram_model_path)
        return ngram_models, ngram_models_rev

    def play_game(self, actual_word, player=None):
        if player:
            self.player = player
        obscured_word = '_' * len(actual_word)
        lives = self.max_lives
        already_guessed_letters = set()
        while lives > 0 and obscured_word != actual_word:
            guessed_letter = self.player.guess_letter(obscured_word, already_guessed_letters)
            already_guessed_letters.add(guessed_letter)
            if guessed_letter in actual_word: 
                indices = [i for i, letter in enumerate(actual_word) if letter == guessed_letter]
                obscured_word_list = list(obscured_word)
                for i in indices:
                    obscured_word_list[i] = guessed_letter
                obscured_word = ''.join(obscured_word_list)
            else:
                lives -= 1
        return obscured_word == actual_word

    def simulate_games(self, n=None):
        if n:
            print(f"using n_{n}_grams")
            word_length_to_n = self.create_word_length_to_n(n)
            print(word_length_to_n)
            self.player = GreedyPlayer(word_length_to_n, self.ngram_models, self.ngram_models_rev)
        else:
            print(self.word_length_to_n)
        num_wins = 0
        total_games = 0
        results_by_length = defaultdict(lambda: {'wins': 0, 'total': 0})
        for i, actual_word in tqdm(enumerate(self.word_list)):
            word_length = len(actual_word)
            if self.play_game(actual_word):
                num_wins += 1
                results_by_length[word_length]['wins'] += 1
            results_by_length[word_length]['total'] += 1
            total_games += 1
        return num_wins, total_games, results_by_length

if __name__ == "__main__":
    # Example usage
    word_list_path = "data/test/words_test.txt"
    models_dir = "results/models"
    num_games = 1000
    game_simulator = GameSimulator(word_list_path, models_dir, max_lives=6, num_games=1000)
    num_wins, total_games, results_by_length = game_simulator.simulate_games()
    print(f"Number of games won: {num_wins} / {total_games}")
    print("results_by_length:")
    print(results_by_length)