# Game Simulator Documentation

## Overview

The `GameSimulator` class is designed to simulate a series of word-guessing games using a pre-trained N-gram model. The simulator plays the games with a greedy guessing strategy, tracks the results, and provides statistics such as win rates for different word lengths.

## Classes and Methods

### `GameSimulator`

The `GameSimulator` class handles the entire game simulation process, from loading the word list and models to running the simulations and exporting the results.

#### **`__init__(self, word_list_path, models_dir, max_lives=6, num_games=1)`**

- **Parameters:**
  - `word_list_path` (str): Path to the file containing the list of words to be used in the simulation.
  - `models_dir` (str): Directory containing the pre-trained N-gram models.
  - `max_lives` (int): The maximum number of incorrect guesses allowed before the game is lost (default is 6).
  - `num_games` (int): The number of games to simulate (default is 1).

- **Functionality:**
  - Initializes the game simulator by loading the word list and the N-gram models.
  - Sets up the `GreedyPlayer` object, which is used to guess letters during the game.

#### **`get_word_length_to_n(self)`**

- **Returns:**
  - A dictionary mapping word lengths to the `n` value to be used by the N-gram model during guessing.

- **Functionality:**
  - This function sets up a mapping that determines which N-gram size to use based on the length of the word. The current implementation uses N-gram size 6 for most word lengths.

#### **`create_word_length_to_n(self, n)`**

- **Parameters:**
  - `n` (int): The N-gram size to be used for all word lengths.

- **Returns:**
  - A dictionary mapping all word lengths (1 to 49) to the specified `n` value.

- **Functionality:**
  - This function creates a uniform mapping where every word length is associated with the same N-gram size, `n`.

#### **`load_all_kneser_ney_models(self, models_dir)`**

- **Parameters:**
  - `models_dir` (str): Directory containing the pre-trained N-gram models.

- **Returns:**
  - A dictionary containing the loaded N-gram models for each N-gram size.

- **Functionality:**
  - This function loads the pre-trained N-gram models (with Kneser-Ney smoothing) from the specified directory. The models are loaded into a dictionary where the keys are the N-gram sizes (3, 4, 5, 6).

#### **`play_game(self, actual_word, player=None)`**

- **Parameters:**
  - `actual_word` (str): The word to be guessed during the game.
  - `player` (`GreedyPlayer`, optional): The player object to be used for guessing. If not provided, the default player is used.

- **Returns:**
  - `bool`: `True` if the word was successfully guessed, `False` otherwise.

- **Functionality:**
  - Simulates a single game where the player tries to guess the given word. The game continues until either the word is guessed correctly or the player runs out of lives. The guessed letters and the current state of the word are updated at each step.

#### **`simulate_games(self, n=None, output_csv_path=None)`**

- **Parameters:**
  - `n` (int, optional): If provided, the simulator will use this N-gram size for all word lengths.
  - `output_csv_path` (str, optional): Path to save the simulation results as a CSV file.

- **Returns:**
  - `num_wins` (int): The total number of games won.
  - `total_games` (int): The total number of games played.
  - `results_by_length` (dict): A dictionary storing the number of wins and total games for each word length.
  - `results_df` (`pandas.DataFrame`): A DataFrame containing the win rates and totals for each word length, sorted by win rate.

- **Functionality:**
  - Simulates a series of games and tracks the number of wins and total games for each word length.
  - Converts the results to a pandas DataFrame, sorts it by win rate in descending order, and appends a total row.
  - If a CSV path is provided, the results are saved to the specified file.

### Example Usage

```python
if __name__ == "__main__":
    word_list_path = "data/test/words_test.txt"
    models_dir = "results/models"
    num_games = 1000
    game_simulator = GameSimulator(word_list_path, models_dir, max_lives=6, num_games=num_games)
    num_wins, total_games, results_by_length, results_df = game_simulator.simulate_games(output_csv_path='game_results.csv')
    print(f"Number of games won: {num_wins} / {total_games}")
    print("results_by_length:")
    print(results_by_length)
```