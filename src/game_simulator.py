from data_preparation import load_data, preprocess_data
from player_agent import GreedyPlayer
from collections import defaultdict
import pickle
import os
from tqdm import tqdm
import pandas as pd

class GameSimulator:
    def __init__(self, word_list_path, models_dir, max_lives=6, num_games=1):
        self.word_list_path = word_list_path
        self.max_lives = max_lives
        self.num_games = num_games
        self.word_list = preprocess_data(load_data(word_list_path))
        #self.ngram_models, self.ngram_models_rev = self.load_all_models(models_dir)
        self.ngram_models_kneser_ney = self.load_all_kneser_ney_models(models_dir)
        self.word_length_to_n = self.get_word_length_to_n()
        self.player = GreedyPlayer(self.word_length_to_n, self.ngram_models_kneser_ney)
        
    def get_word_length_to_n(self):
        word_length_to_n = {}
        for l in range(1, 50):
            if l in range(6, 50):
                word_length_to_n[l] = 6
            elif l in range(4, 6):
                word_length_to_n[l] = 6
            elif l in [3]:
                word_length_to_n[l] = 4
            else:
                word_length_to_n[l] = 3
        return word_length_to_n
    
    def create_word_length_to_n(self, n):
        word_length_to_n = {}
        for l in range(1, 50):
            word_length_to_n[l] = n
        return word_length_to_n

    def load_all_kneser_ney_models(self, models_dir):
        n_values = [3, 4, 5, 6]
        ngram_models_kneser_ney = {}
        for n in n_values:
            model_name = f"n_{n}_gram_model_kneser_ney.pkl"
            model_path = os.path.join(models_dir, model_name)
            with open(model_path, 'rb') as file:
                ngram_models_kneser_ney[n] = pickle.load(file)
                print("loading", model_path)
        return ngram_models_kneser_ney

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

    def simulate_games(self, n=None, output_csv_path=None):
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
            #if word_length > 7:
            #    continue
            if self.play_game(actual_word):
                num_wins += 1
                results_by_length[word_length]['wins'] += 1
            results_by_length[word_length]['total'] += 1
            total_games += 1

        # Convert results_by_length to a pandas DataFrame
        results_df = pd.DataFrame([
            {'length': length, 'total': result['total'], 'wins': result['wins'], 
             'win_rate': round((result['wins'] / result['total']) * 100, 2) if result['total'] > 0 else 0}
            for length, result in results_by_length.items()
        ])

        # Sort by win_rate descending
        results_df = results_df.sort_values(by='win_rate', ascending=False)

        # Calculate total row
        total_row = pd.DataFrame({
            'length': ['Total'],
            'total': [results_df['total'].sum()],
            'wins': [results_df['wins'].sum()],
            'win_rate': [round((results_df['wins'].sum() / results_df['total'].sum()) * 100, 2)]
        })

        # Append the total row to the DataFrame using pd.concat
        results_df = pd.concat([results_df, total_row], ignore_index=True)
        
        # Save DataFrame to CSV if a path is provided
        if output_csv_path:
            results_df.to_csv(output_csv_path, index=False)
            print(f"Results saved to {output_csv_path}")

        return num_wins, total_games, results_by_length, results_df

if __name__ == "__main__":
    # Example usage
    word_list_path = "data/test/words_test.txt"
    models_dir = "results/models"
    num_games = 1000
    game_simulator = GameSimulator(word_list_path, models_dir, max_lives=6, num_games=1000)
    num_wins, total_games, results_by_length, results_df = game_simulator.simulate_games(output_csv_path='game_results.csv')
    print(f"Number of games won: {num_wins} / {total_games}")
    print("results_by_length:")
    print(results_by_length)