from data_preparation import load_data, preprocess_data
from player_agent import GreedyPlayer
import random

class GameSimulator:
    def __init__(self, word_list_path, n, model_path, max_lives=5, num_games=100):
        self.word_list_path = word_list_path
        self.model_path = model_path
        self.max_lives = max_lives
        self.num_games = num_games
        self.word_list = preprocess_data(load_data(word_list_path))
        self.player = GreedyPlayer(n, model_path)

    def play_game(self, actual_word):
        obscured_word = '_' * len(actual_word)
        lives = self.max_lives
        while lives > 0 and obscured_word != actual_word:
            guessed_letter = self.player.guess_letter(obscured_word)
            if guessed_letter in actual_word:
                # Update the obscured word with the guessed letter
                indices = [i for i, letter in enumerate(actual_word) if letter == guessed_letter]
                obscured_word_list = list(obscured_word)
                for i in indices:
                    obscured_word_list[i] = guessed_letter
                obscured_word = ''.join(obscured_word_list)
            else:
                lives -= 1

        return obscured_word == actual_word

    def simulate_games(self):
        num_wins = 0
        for i in range(self.num_games):
            print("game", i)
            actual_word = random.choice(self.word_list)
            if self.play_game(actual_word):
                num_wins += 1
        return num_wins

if __name__ == "__main__":
    # Example usage
    word_list_path = "../data/test/words_test.txt"
    n = 3
    model_path = f"../results/models/n_{n}_gram_model.pkl"
    game_simulator = GameSimulator(word_list_path, n, model_path)
    num_wins = game_simulator.simulate_games()
    print(f"Number of games won: {num_wins}")