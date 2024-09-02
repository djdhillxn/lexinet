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


# Training

## Forward

1.	**Purpose**: The train_gold method trains an N-gram model with additional masking techniques to handle cases where parts of the N-gram are unknown or need to be generalized.

2.	**Padded Word**: Each word is padded with special tokens (`'<s>'` and `'</s>'`) to account for the start and end of the word. The padding ensures that the model learns how words typically begin and end. The padding length is n-1 on both sides.

3.	**N-gram Generation**: The method generates N-grams (sequences of n consecutive items) from the padded word. Each N-gram is split into a prefix (all items except the last) and a suffix (the last item).

4.	**Skipping End Tokens**: If the suffix is the end token `'</s>'`, it is skipped, as the model doesn’t need to predict an end token.

5.	**Masking Technique**:

    •	The method identifies positions within the N-gram that are not padding tokens (`'<s>'` or `'</s>'`).

    •	It creates masked versions of the N-gram by replacing some of these positions with an underscore ('_'). This process is done iteratively for different combinations of masked positions.

    •	The number of positions masked is determined by the cnt-1 and n-2 logic, ensuring that at least one position remains unmasked in N-grams longer than 2.

6.	**Updating Counts**: The counts of these N-grams and their masked variants are updated in the self.ngrams dictionary. This helps the model learn to generalize when certain parts of the N-gram are missing or uncertain.

## Backward

1.	**Purpose**: The train_reverse_gold method is similar to train_gold, but it trains a reverse N-gram model. This model predicts the preceding character in a word given the succeeding context.

2.	**Padded Word**: As in train_gold, each word is padded with `'<s>'` and `'</s>'` to handle the start and end of the word.

3.	**N-gram Generation**: N-grams are generated, but here the prefix is the first item in the N-gram, and the suffix is the rest of the sequence.

4.	**Skipping Start Tokens**: If the prefix is the start token `'<s>'`, it is skipped, as the model doesn’t need to predict anything before the start of a word.

5.	**Masking Technique**:

    -   Similar to train_gold, positions within the suffix that are not padding tokens are identified.

    -	Masked versions of the suffix are generated by replacing some positions with an underscore.

    -	This helps the model handle cases where certain parts of the context might be unknown.

6.	**Updating Counts**: The counts of these reverse N-grams and their masked variants are updated in the self.ngrams_rev dictionary. This allows the model to predict the prefix (preceding letter) based on the following context.