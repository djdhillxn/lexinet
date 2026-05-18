from src.data_preparation import load_data, preprocess_data
from src.player_agent import GreedyPlayer
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import pickle
import os
import multiprocessing as mp
from tqdm import tqdm

_WORKER_PLAYER = None
_WORKER_MAX_LIVES = None


def _load_all_kneser_ney_models(models_dir, run_name, verbose=True):
    n_values = [3, 4, 5, 6]
    ngram_models_kneser_ney = {}
    for n in n_values:
        model_name = f"n_{n}_gram_model_{run_name}.pkl"
        model_path = os.path.join(models_dir, model_name)
        with open(model_path, 'rb') as file:
            ngram_models_kneser_ney[n] = pickle.load(file)
            if verbose:
                print("loading", model_path)
    return ngram_models_kneser_ney


def _play_game_with_player(actual_word, player, max_lives):
    obscured_word = '_' * len(actual_word)
    lives = max_lives
    already_guessed_letters = set()
    while lives > 0 and obscured_word != actual_word:
        guessed_letter = player.guess_letter(obscured_word, already_guessed_letters)
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


def _initialize_worker(models_dir, run_name, word_length_to_n, max_lives, method_name):
    global _WORKER_PLAYER
    global _WORKER_MAX_LIVES

    ngram_models_kneser_ney = _load_all_kneser_ney_models(models_dir, run_name, verbose=False)
    _WORKER_PLAYER = GreedyPlayer(word_length_to_n, ngram_models_kneser_ney, method_name)
    _WORKER_MAX_LIVES = max_lives


def _simulate_word_in_worker(actual_word):
    if _WORKER_PLAYER is None or _WORKER_MAX_LIVES is None:
        raise RuntimeError("Worker was not initialized before simulating games.")

    won = _play_game_with_player(actual_word, _WORKER_PLAYER, _WORKER_MAX_LIVES)
    return len(actual_word), won


class GameSimulator:
    def __init__(self, word_list_path, models_dir, run_name='kneser_ney', method_name='best', max_lives=6, num_games=1):
        self.word_list_path = word_list_path
        self.max_lives = max_lives
        self.num_games = num_games
        self.models_dir = models_dir
        self.run_name = run_name
        self.method_name = method_name
        self.word_list = preprocess_data(load_data(word_list_path))
        self.word_length_to_n = self.get_word_length_to_n()
        self.ngram_models_kneser_ney = _load_all_kneser_ney_models(self.models_dir, self.run_name, verbose=False)
        self.player = GreedyPlayer(self.word_length_to_n, self.ngram_models_kneser_ney, self.method_name)
        
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

    def _create_results_df(self, results_by_length):
        import pandas as pd

        results_df = pd.DataFrame([
            {'length': length, 'total': result['total'], 'wins': result['wins'],
             'win_rate': round((result['wins'] / result['total']) * 100, 2) if result['total'] > 0 else 0}
            for length, result in results_by_length.items()
        ])

        results_df = results_df.sort_values(by='win_rate', ascending=False)

        total_row = pd.DataFrame({
            'length': ['Total'],
            'total': [results_df['total'].sum()],
            'wins': [results_df['wins'].sum()],
            'win_rate': [round((results_df['wins'].sum() / results_df['total'].sum()) * 100, 2)]
        })

        return pd.concat([results_df, total_row], ignore_index=True)

    def _save_results_if_needed(self, results_df, output_csv_path):
        if output_csv_path:
            results_df.to_csv(output_csv_path, index=False)
            print(f"Results saved to {output_csv_path}")

    def _simulate_games_parallel(self, word_length_to_n, n_workers, chunksize, multiprocessing_start_method):
        global _WORKER_PLAYER
        global _WORKER_MAX_LIVES

        num_wins = 0
        total_games = 0
        results_by_length = defaultdict(lambda: {'wins': 0, 'total': 0})

        if chunksize is None:
            chunksize = max(1, len(self.word_list) // (n_workers * 8))

        context = mp.get_context(multiprocessing_start_method) if multiprocessing_start_method else mp.get_context()
        start_method = context.get_start_method()
        executor_kwargs = {"max_workers": n_workers, "mp_context": context}

        if start_method == "fork":
            _WORKER_PLAYER = GreedyPlayer(word_length_to_n, self.ngram_models_kneser_ney, self.method_name)
            _WORKER_MAX_LIVES = self.max_lives
        else:
            executor_kwargs["initializer"] = _initialize_worker
            executor_kwargs["initargs"] = (self.models_dir, self.run_name, word_length_to_n, self.max_lives)

        try:
            with ProcessPoolExecutor(**executor_kwargs) as executor:
                results = executor.map(_simulate_word_in_worker, self.word_list, chunksize=chunksize)
                for word_length, won in tqdm(results, total=len(self.word_list)):
                    if won:
                        num_wins += 1
                        results_by_length[word_length]['wins'] += 1
                    results_by_length[word_length]['total'] += 1
                    total_games += 1
        finally:
            if start_method == "fork":
                _WORKER_PLAYER = None
                _WORKER_MAX_LIVES = None

        return num_wins, total_games, results_by_length

    def simulate_games(self, n=None, output_csv_path=None, n_workers=1, chunksize=None, multiprocessing_start_method=None):
        if n:
            print(f"using n_{n}_grams")
            word_length_to_n = self.create_word_length_to_n(n)
            print(word_length_to_n)
            self.player = GreedyPlayer(word_length_to_n, self.ngram_models_kneser_ney, self.method_name)
        else:
            word_length_to_n = self.word_length_to_n
            print(self.word_length_to_n)

        if n_workers is None:
            n_workers = os.cpu_count() or 1
        if n_workers < 1:
            raise ValueError("n_workers must be None or a positive integer.")

        if n_workers > 1:
            num_wins, total_games, results_by_length = self._simulate_games_parallel(
                word_length_to_n,
                n_workers,
                chunksize,
                multiprocessing_start_method,
            )
            results_df = self._create_results_df(results_by_length)
            self._save_results_if_needed(results_df, output_csv_path)
            return num_wins, total_games, results_by_length, results_df
        print("num workers 1 code is running successfully right...")
        num_wins = 0
        total_games = 0
        results_by_length = defaultdict(lambda: {'wins': 0, 'total': 0})
        for i, actual_word in tqdm(enumerate(self.word_list)):
            word_length = len(actual_word)
            #if word_length > 7:
            #    continue
            if _play_game_with_player(actual_word, self.player, self.max_lives):
                num_wins += 1
                results_by_length[word_length]['wins'] += 1
            results_by_length[word_length]['total'] += 1
            total_games += 1

        # Convert results_by_length to a pandas DataFrame
        import pandas as pd

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
