import random
from collections import defaultdict, Counter
import pickle
import os

class GreedyPlayer:
    def __init__(self, word_length_to_n, ngram_models_kneser_ney_dict):
        self.k = 0.05 #0.05 
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"
        self.word_length_to_n = word_length_to_n
        self.ngram_models = ngram_models_kneser_ney_dict

    def guess_letter(self, known_word, already_guessed_letters, override=False):
        word_length = len(known_word)
        n = self.word_length_to_n.get(word_length, 2)
        padded_word = ['<s>'] * (n - 1) + list(known_word) + ['</s>'] * (n - 1)
        known_letters = {ch for ch in known_word if ch != '_'}
        known_letters = known_letters.union(already_guessed_letters)
        alphabet = set(self.alphabet) - known_letters
        candidates = defaultdict(float)
        for i in range(n-1, len(padded_word) - (n-1)):
            if padded_word[i] == '_':
                for letter in alphabet:  
                    if word_length > 9:
                        forward_prob, reverse_prob = self.calculate_probability(padded_word, i, n, letter)
                        combined_prob = forward_prob * reverse_prob
                        candidates[letter] += combined_prob
                    else:
                        prefix_fwd = tuple(padded_word[i-(n-1):i])
                        forward_prob = self.calculate_forward_probability(prefix_fwd, letter, n)
                        suffix_rev = tuple(padded_word[i+1:i+(n-1)+1])
                        reverse_prob = self.calculate_backward_probability(suffix_rev, letter, n)
                        combined_prob = forward_prob * reverse_prob
                        candidates[letter] += combined_prob
        
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        if sorted_candidates:
            best_letter = sorted_candidates[0][0]
        else:
            best_letter = random.choice(list(alphabet))
        
        return best_letter

    def calculate_probability(self, padded_word, i, n, letter):
        # Forward probability
        suffix = letter
        prefix = tuple(padded_word[i-(n-1):i])
        forward_prob = 0
        ngrams = self.ngram_models[n]['ngrams']
        ngrams_rev = self.ngram_models[n]['ngrams_rev']
        if prefix in ngrams:
            forward_count = ngrams[prefix][suffix] + self.k
            total_count = sum(ngrams[prefix].values()) + self.k * len(self.alphabet)
            forward_prob = forward_count / total_count
        elif self.k != 0:
            forward_prob = self.k / (self.k * len(self.alphabet))

        # Reverse probability
        prefix = letter
        suffix = tuple(padded_word[i+1:i+(n-1)+1])
        reverse_prob = 0
        if suffix in ngrams_rev:
            reverse_count = ngrams_rev[suffix][prefix] + self.k
            total_count = sum(ngrams_rev[suffix].values()) + self.k * len(self.alphabet)
            reverse_prob = reverse_count / total_count
        elif self.k != 0:
            reverse_prob = self.k / (self.k * len(self.alphabet))

        return forward_prob, reverse_prob

    def calculate_forward_probability(self, prefix, suffix, n):
        forward_prob = 0
        ngrams = self.ngram_models[n]['ngrams']
        if prefix in ngrams:
            forward_count = ngrams[prefix][suffix] #+ self.k
            total_count = sum(ngrams[prefix].values()) #+ self.k * len(self.alphabet)
            forward_prob = forward_count / total_count
        #elif self.k != 0:
        #    forward_prob = 0.001 #self.k / (self.k * len(self.alphabet))
        mu = 0.80
        if len(prefix) > 2:
            backoff_prob = self.calculate_forward_probability(prefix[1:], suffix, n - 1)
            forward_prob = mu * forward_prob + (1 - mu) * backoff_prob
        
        return forward_prob

    def calculate_backward_probability(self, suffix, prefix, n):
        reverse_prob = 0
        ngrams_rev = self.ngram_models[n]['ngrams_rev']
        if suffix in ngrams_rev:
            reverse_count = ngrams_rev[suffix][prefix] #+ self.k
            total_count = sum(ngrams_rev[suffix].values()) #+ self.k * len(self.alphabet)
            reverse_prob = reverse_count / total_count
        #elif self.k != 0:
        #    reverse_prob = 0.001# / (self.k * len(self.alphabet))
        mu = 0.80
        if len(suffix) > 2:
            backoff_prob = self.calculate_backward_probability(suffix[:-1], prefix, n - 1)
            reverse_prob = mu * reverse_prob + (1 - mu) * backoff_prob

        return reverse_prob

    def kneser_ney_probability_old(self, prefix, suffix, n):
        D = 0.75  # Discount value

        ngrams = self.ngram_models[n]['ngrams']
        continuation_counts = self.ngram_models[n]['continuation_counts']
        unigrams = self.ngram_models[n]['unigrams']

        if prefix in ngrams:
            prefix_count = sum(ngrams[prefix].values())
            suffix_count = ngrams[prefix][suffix]
            continuation_count = len(continuation_counts[suffix])
            
            if prefix_count > 0:
                lambda_prefix = (D / prefix_count) * len(ngrams[prefix])
                if suffix_count > 0:
                    return max(suffix_count - D, 0) / prefix_count + lambda_prefix * continuation_count / sum(unigrams.values())
                else:
                    return lambda_prefix * continuation_count / sum(unigrams.values())
            else:
                # Handle case where prefix_count is zero (optional, depending on your logic)
                return 1 / 26  # Or any default value that fits your application logic
        else:
            # Handle case where prefix is not found in ngrams (optional, depending on your logic)
            return 1 / 26  # Or any default value that fits your application logic

    def kneser_ney_probability(self, prefix, suffix, n):
        D = 0.75  # Discount value

        ngrams = self.ngram_models[n]['ngrams']
        continuation_counts = self.ngram_models[n]['continuation_counts']
        unigrams = self.ngram_models[n]['unigrams']
        
        prefix_count = sum(ngrams[prefix].values())
        suffix_count = ngrams[prefix][suffix]
        continuation_count = len(continuation_counts[suffix])
        
        prob = 0

        if prefix_count > 0:
            lambda_prefix = (D / prefix_count) * len(ngrams[prefix])
            prob = max(suffix_count - D, 0) / prefix_count + lambda_prefix * continuation_count / sum(unigrams.values())
        else:
            prob = 1 / 26  # Or any default value that fits your application logic
        
        mu = 0.8
        if len(prefix) > 2:
            backoff_prob = self.kneser_ney_probability(prefix[1:], suffix, n-1)
            prob = mu * prob + (1 - mu) * backoff_prob
        
        return prob

    def kneser_ney_probability_reverse(self, suffix, prefix, n):
        D = 0.75  # Discount value
        
        ngrams_rev = self.ngram_models[n]['ngrams_rev']
        continuation_counts_rev = self.ngram_models[n]['continuation_counts_rev']
        unigrams_rev = self.ngram_models[n]['unigrams_rev']

        # set self.ngrams_rev, self.continuation_counts_rev, self.unigrams_rev to get for n=n grams 
        suffix_count = sum(ngrams_rev[suffix].values())
        prefix_count = ngrams_rev[suffix][prefix]
        continuation_count = len(continuation_counts_rev[prefix])
        
        prob = 0

        if suffix_count > 0:
            lambda_suffix = (D / suffix_count) * len(ngrams_rev[suffix])
            prob = max(prefix_count - D, 0) / suffix_count + lambda_suffix * continuation_count / sum(unigrams_rev.values())
        else:
            prob = 1 / 26  # Or any default value that fits your application logic
        
        mu = 0.8
        if len(suffix) > 2:
            backoff_prob = self.kneser_ney_probability_reverse(suffix[:-1], prefix, n-1)
            prob = mu * prob + (1 - mu) * backoff_prob
        
        return prob

    def kneser_ney_probability_reverse_old(self, suffix, prefix, n):
        D = 0.75  # Discount value

        ngrams_rev = self.ngram_models[n]['ngrams_rev']
        continuation_counts_rev = self.ngram_models[n]['continuation_counts_rev']
        unigrams_rev = self.ngram_models[n]['unigrams_rev']

        suffix_count = sum(ngrams_rev[suffix].values())
        prefix_count = ngrams_rev[suffix][prefix]
        continuation_count = len(continuation_counts_rev[prefix])
        
        if suffix_count > 0:
            lambda_suffix = (D / suffix_count) * len(ngrams_rev[suffix])
            if prefix_count > 0:
                return max(prefix_count - D, 0) / suffix_count + lambda_suffix * continuation_count / sum(unigrams_rev.values())
            else:
                return lambda_suffix * continuation_count / sum(unigrams_rev.values())
        else:
            # Handle case where suffix_count is zero (optional, depending on your logic)
            return 1 / 26  # Or any default value that fits your application logic

    def guess_letter_kneser(self, known_word, already_guessed_letters):
        word_length = len(known_word)
        n = 6 #self.word_length_to_n.get(word_length, 2)
        padded_word = ['<s>'] * (n - 1) + list(known_word) + ['</s>'] * (n - 1)
        known_letters = {ch for ch in known_word if ch != '_'}
        known_letters = known_letters.union(already_guessed_letters)
        alphabet = set(self.alphabet) - known_letters
        candidates = defaultdict(float)
        #if len(alphabet) == 26:
        #    return 'e'
        
        for i in range(n-1, len(padded_word) - (n-1)):
            if padded_word[i] == '_':
                for letter in alphabet:
                    padded_word[i] = letter
                    prefix = tuple(padded_word[i-n+1:i])
                    suffix = padded_word[i]
                    forward_prob = self.kneser_ney_probability(prefix, suffix, n)

                    suffix_rev = tuple(padded_word[i+1:i+n])
                    prefix_rev = padded_word[i]
                    reverse_prob = self.kneser_ney_probability_reverse(suffix_rev, prefix_rev, n)

                    combined_prob = forward_prob * reverse_prob
                    candidates[letter] += combined_prob
                    padded_word[i] = '_'
        
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        if sorted_candidates:
            best_letter = sorted_candidates[0][0]
        else:
            best_letter = random.choice(list(alphabet))
        
        return best_letter

    def guess_letter_bestest(self, known_word, already_guessed_letters, override=False):
        word_length = len(known_word)
        n = self.word_length_to_n.get(word_length, 2)
        padded_word = ['<s>'] * (n - 1) + list(known_word) + ['</s>'] * (n - 1)
        #if word_length <= 50:    
        #    return self.guess_letter_kneser(known_word, already_guessed_letters)
        known_letters = {ch for ch in known_word if ch != '_'}
        known_letters = known_letters.union(already_guessed_letters)
        alphabet = set(self.alphabet) - known_letters
        candidates = defaultdict(float)
        #if len(alphabet) == 26:
        #    return 'e'
        
        for i in range(n-1, len(padded_word) - (n-1)):
            if padded_word[i] == '_':
                for letter in alphabet:  
                    if word_length > 9:
                        forward_prob, reverse_prob = self.calculate_probability(padded_word, i, n, letter)
                        combined_prob = forward_prob * reverse_prob
                        candidates[letter] += combined_prob
                    else:
                        prefix_fwd = tuple(padded_word[i-(n-1):i])
                        forward_prob = self.calculate_forward_probability(prefix_fwd, letter, n)
                        suffix_rev = tuple(padded_word[i+1:i+(n-1)+1])
                        reverse_prob = self.calculate_backward_probability(suffix_rev, letter, n)
                        combined_prob = forward_prob * reverse_prob
                        candidates[letter] += combined_prob
        
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        if sorted_candidates:
            best_letter = sorted_candidates[0][0]
        else:
            best_letter = random.choice(list(alphabet))
        
        return best_letter
    
    def guess_letter_orr(self, known_word, already_guessed_letters, override=False):
        word_length = len(known_word)
        n = self.word_length_to_n.get(word_length, 2)
        padded_word = ['<s>'] * (n - 1) + list(known_word) + ['</s>'] * (n - 1)
        # if word_length <= 50:    
        #     return self.guess_letter_kneser(known_word, already_guessed_letters)
        known_letters = {ch for ch in known_word if ch != '_'}
        known_letters = known_letters.union(already_guessed_letters)
        alphabet = set(self.alphabet) - known_letters
        probabilities_matrix = []
        #if len(alphabet) == 26:
        #    return 'e'
        
        for i in range(n-1, len(padded_word) - (n-1)):
            if padded_word[i] == '_':
                letter_probs = []
                for letter in alphabet:  
                    forward_prob, reverse_prob, flags = self.calculate_probability(padded_word, i, n, letter)
                    #prefix_fwd = tuple(padded_word[i-(n-1):i])
                    #forward_prob = self.calculate_forward_probability(prefix_fwd, letter, n)
                    #suffix_rev = tuple(padded_word[i+1:i+(n-1)+1])
                    #reverse_prob = self.calculate_backward_probability(suffix_rev, letter, n)
                    combined_prob = forward_prob * reverse_prob
                    letter_probs.append((letter, combined_prob))

                    
                probabilities_matrix.append(letter_probs)

        # Find the letter with the highest probability in the matrix
        best_letter = None
        max_prob = -1
        for letter_probs in probabilities_matrix:
            for letter, prob in letter_probs:
                if prob > max_prob:
                    max_prob = prob
                    best_letter = letter
        
        if best_letter:
            return best_letter
        else:
            return random.choice(list(alphabet))

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