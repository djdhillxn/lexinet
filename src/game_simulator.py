from data_preparation import load_data, preprocess_data
from player_agent import GreedyPlayer
from collections import defaultdict
import pickle

class GameSimulator:
    def __init__(self, word_list_path, n, model_path, rev_ngram_model_path, max_lives=6, num_games=1):
        self.word_list_path = word_list_path
        #self.model_path = model_path
        self.max_lives = max_lives
        self.num_games = num_games
        self.word_list = preprocess_data(load_data(word_list_path))
        self.ngram_models, self.ngram_models_rev = self.load_all_models()
        #self.player = GreedyPlayer(n, model_path, rev_ngram_model_path)
        #self.word_length_to_n = {length: min(max(2, length // 3), 4) for length in range(2, 21)}
        self.word_length_to_n = self.get_word_length_to_n()
        self.player = GreedyPlayer(n, self.word_length_to_n, self.ngram_models, self.ngram_models_rev)
        #self.player = GreedyPlayer(self.word_length_to_n, self.ngram_models, self.ngram_models_rev)

    def get_word_length_to_n(self):
        word_length_to_n = {}
        for l in range(1, 50):
            if l in range(8, 16):
                word_length_to_n[l] = 4
            else:
                word_length_to_n[l] = 2
        return word_length_to_n

    def load_all_models(self):
        n_values = [2, 3, 4]
        ngram_models = {}
        ngram_models_rev = {}
        for n in n_values:
            model_path = f"results/models/n_{n}_gram_model_n.pkl"
            rev_ngram_model_path = f"results/models/n_{n}_gram_model_rev_n.pkl"
            with open(model_path, 'rb') as file:
                ngram_models[n] = pickle.load(file)
            with open(rev_ngram_model_path, 'rb') as file:
                ngram_models_rev[n] = pickle.load(file)
        
        return ngram_models, ngram_models_rev

    def play_game(self, actual_word):
        obscured_word = '_' * len(actual_word)
        lives = self.max_lives
        #print(lives)
        already_guessed_letters = set()
        while lives > 0 and obscured_word != actual_word:
            #print("guessing...")
            guessed_letter = self.player.guess_letter(obscured_word, already_guessed_letters)
            already_guessed_letters.add(guessed_letter)
            if guessed_letter in actual_word: 
                # Update the obscured word with the guessed letter
                indices = [i for i, letter in enumerate(actual_word) if letter == guessed_letter]
                obscured_word_list = list(obscured_word)
                for i in indices:
                    obscured_word_list[i] = guessed_letter
                obscured_word = ''.join(obscured_word_list)
                #print(obscured_word)
            else:
                lives -= 1
            #print(obscured_word, guessed_letter, lives)

        return obscured_word == actual_word

    def simulate_games_old(self):
        num_wins = 0
        total_games = 0
        from tqdm import tqdm
        for i, actual_word in tqdm(enumerate(self.word_list)):
            #if i == 10000:
            #    break
            #actual_word = random.choice(self.word_list)
            #print("game:", i, "word:", actual_word)
            if self.play_game(actual_word):
                num_wins += 1
            total_games += 1
        return num_wins, total_games
    
    def simulate_games(self):
        num_wins = 0
        total_games = 0
        results_by_length = defaultdict(lambda: {'wins': 0, 'total': 0})
        from tqdm import tqdm
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
    n = 3
    model_path = f"results/models/n_{n}_gram_model_n.pkl"
    rev_ngram_model_path = f"results/models/n_{n}_gram_model_rev_n.pkl"
    print("loading model", model_path)
    print("laoding rev model", rev_ngram_model_path)
    num_games = 1000
    #print("playing number of games:", num_games)
    game_simulator = GameSimulator(word_list_path, n, model_path, rev_ngram_model_path, max_lives=6, num_games=1000)
    num_wins, total_games, results_by_length = game_simulator.simulate_games()
    print(f"Number of games won: {num_wins} / {total_games}")
    print("results_by_length:")
    print(results_by_length)