import random
import pickle

class GreedyPlayer:
    def __init__(self, n, ngram_model_path):
        self.n = n
        self.guessed_letters = set()
        self.ngrams = None
        self.load(ngram_model_path)
        print("gogo")

    def load(self, file_path):
        with open(file_path, 'rb') as file:
            self.ngrams = pickle.load(file)

        self.ngram_model = NgramModel(3)
        self.ngram_model.load(ngram_model_path)
        print("gogo")
        
    def guess_letter(self, obscured_word):
        # Filter out already guessed letters
        available_letters = [letter for letter in obscured_word if letter not in self.guessed_letters]

        # Guess the most probable letter based on n-gram model
        max_prob = 0
        best_letter = None
        for letter in available_letters:
            prefix = obscured_word.replace('_', '')[-(self.ngram_model.n - 1):]
            prob = self.ngrams.get_letter_probability(prefix, letter)
            if prob > max_prob:
                max_prob = prob
                best_letter = letter

        # If no letter was found, guess a random letter
        if best_letter is None:
            best_letter = random.choice(available_letters)

        # Add the guessed letter to the set of guessed letters
        self.guessed_letters.add(best_letter)
        return best_letter

    def reset_guessed_letters(self):
        self.guessed_letters = set()