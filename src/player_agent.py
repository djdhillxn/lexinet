import random
import pickle
from collections import defaultdict, Counter

class GreedyPlayer:
    def __init__(self, n, word_length_to_n, ngram_models_dict, ngram_models_rev_dict):
        self.n = n
        self.k = 0.05
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"
        self.word_length_to_n = word_length_to_n
        self.ngram_models = ngram_models_dict
        self.ngram_models_rev = ngram_models_rev_dict

    def guess_letter(self, known_word, already_guessed_letters):
        word_length = len(known_word)
        n = self.word_length_to_n.get(word_length, 2)
        self.ngrams = self.ngram_models[n]
        self.ngrams_rev = self.ngram_models_rev[n]
        padded_word = ['<s>'] * (n - 1) + list(known_word) + ['</s>']
        known_letters = {ch for ch in known_word if ch != '_'}
        known_letters = known_letters.union(already_guessed_letters)
        alphabet = set(self.alphabet) - known_letters
        candidates = defaultdict(float)
        if len(alphabet) == 26:
            return 'e'

        for i in range(n-1, len(padded_word) - (n-1)):
            if padded_word[i] == '_':
                for letter in alphabet:
                    # Forward probability
                    suffix = letter
                    prefix = tuple(padded_word[i-(n-1):i])
                    forward_prob = 0
                    if prefix in self.ngrams:
                        forward_count = self.ngrams[prefix][suffix] + self.k
                        total_count = sum(self.ngrams[prefix].values()) + self.k * len(alphabet)
                        forward_prob = forward_count / total_count
                    else:
                        forward_prob = self.k / (self.k * len(alphabet))

                    # Reverse probability
                    prefix = letter
                    suffix = tuple(padded_word[i+1:i+(n-1)+1])
                    reverse_prob = 0
                    if suffix in self.ngrams_rev:
                        reverse_count = self.ngrams_rev[suffix][prefix] + self.k
                        total_count = sum(self.ngrams_rev[suffix].values()) + self.k * len(alphabet)
                        reverse_prob = reverse_count / total_count
                    else:
                        reverse_prob = self.k / (self.k * len(alphabet))

                    #candidates[letter] += forward_prob + reverse_prob gives 52712/170671 for n=4 gram
                    candidates[letter] += forward_prob * reverse_prob

        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        if sorted_candidates:
            best_letter = sorted_candidates[0][0]
        else:
            best_letter = random.choice(list(alphabet))  # If no candidates, guess a random letter

        return best_letter

    """
    def guess_letter_old(self, obscured_word, already_guessed_letters):
        available_letters = [letter for letter in self.alphabet if letter not in already_guessed_letters]
        #print(len(available_letters))
        # Guess the most probable letter based on n-gram model
        max_prob = 0
        best_letter = None
        for letter in available_letters:
            prefix = obscured_word.replace('_', '')[-(self.n - 1):]
            #print(prefix)
            prob = self.ngrams[prefix][letter]
            if prob > max_prob:
                max_prob = prob
                best_letter = letter

        # If no letter was found, guess a random letter
        if best_letter is None:
            best_letter = random.choice(available_letters)

        return best_letter

    def guess_letter_v1(self, known_word, already_guessed_letters):
        known_letters = {ch for ch in known_word if ch != '_'}
        known_letters = known_letters.union(already_guessed_letters)
        alphabet = set(self.alphabet) - known_letters
        candidates = defaultdict(float)
        
        if len(alphabet) == 26:  # No letters guessed yet, return 'e' as a common starting point
            return 'e'

        # Create candidate contexts
        for i in range(len(known_word)):
            if known_word[i] == '_':
                for letter in alphabet:
                    context_before = known_word[max(0, i-self.n+1):i]
                    context_after = known_word[i+1:min(len(known_word), i+self.n-1)]
                    context = tuple(context_before) + (letter,) + tuple(context_after)
                    
                    # Calculate probability of the letter given the context
                    prefix = context[:-1]
                    suffix = context[-1]
                    if prefix in self.ngrams:
                        prob = self.ngrams[prefix][suffix] / sum(self.ngrams[prefix].values())
                        candidates[letter] += prob
        
        # Sort candidates by probability
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_candidates:
            best_letter = sorted_candidates[0][0]
        else:
            best_letter = random.choice(list(alphabet))  # If no candidates, guess a random letter

        return best_letter
    
    def guess_letter_o(self, known_word, already_guessed_letters):
        # Pad the known word
        padded_word = ['<s>'] * (self.n - 1) + list(known_word) + ['</s>']

        known_letters = {ch for ch in known_word if ch != '_'}
        known_letters = known_letters.union(already_guessed_letters)
        alphabet = set(self.alphabet) - known_letters
        candidates = defaultdict(float)
        
        if len(alphabet) == 26:  # No letters guessed yet, return 'e' as a common starting point
            return 'e'

        # Create candidate contexts
        for i in range(self.n - 1, len(padded_word) - (self.n - 1)):
            if padded_word[i] == '_':
                for letter in alphabet:
                    context_before = padded_word[max(0, i-self.n+1):i]
                    context_after = padded_word[i+1:min(len(padded_word), i+self.n-1)]
                    context = tuple(context_before) + (letter,) + tuple(context_after)
                    
                    # Calculate probability of the letter given the context
                    prefix = context[:-1]
                    suffix = context[-1]
                    prob = 0
                    if prefix in self.ngrams:
                        prob = self.ngrams[prefix][suffix] / sum(self.ngrams[prefix].values())
                        candidates[letter] += prob
                        print(f"Context: {context}, Prefix: {prefix}, Suffix: {suffix}, Prob: {prob}")
        
        # Sort candidates by probability
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        print(f"Candidates: {sorted_candidates}")
        
        if sorted_candidates:
            best_letter = sorted_candidates[0][0]
        else:
            best_letter = random.choice(list(alphabet))  # If no candidates, guess a random letter

        return best_letter
    
    def guess_letter_brr(self, known_word, already_guessed_letters):
        # Pad the known word
        padded_word = ['<s>'] * (self.n - 1) + list(known_word) + ['</s>']
        known_letters = {ch for ch in known_word if ch != '_'}
        known_letters = known_letters.union(already_guessed_letters)
        alphabet = set(self.alphabet) - known_letters
        candidates = defaultdict(float)
        
        if len(alphabet) == 26:  # No letters guessed yet, return 'e' as a common starting point
            return 'e'

        # Create candidate contexts
        for i in range(self.n - 1, len(padded_word) - (self.n - 1)):
            if padded_word[i] == '_':
                for letter in alphabet:
                    context_before = padded_word[max(0, i-self.n+1):i-1]
                    context_after = padded_word[i+1:min(len(padded_word), i+self.n-1)]
                    context = tuple(context_before) + (letter,) + tuple(context_after)
                    print(context)
                    # Calculate probability of the letter given the context
                    prefix = context[:-1]
                    suffix = context[-1]
                    prob = 0
                    if prefix in self.ngrams:
                        count = self.ngrams[prefix][suffix]
                        prob = self.ngrams[prefix][suffix] / sum(self.ngrams[prefix].values())
                        candidates[letter] += count
                        print(f"Context: {context}, Prefix: {prefix}, Suffix: {suffix}, Count: {count}")
        
        # Sort candidates by probability
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        print(f"Candidates: {sorted_candidates}")
        
        if sorted_candidates:
            best_letter = sorted_candidates[0][0]
        else:
            best_letter = random.choice(list(alphabet))  # If no candidates, guess a random letter

        return best_letter
    """