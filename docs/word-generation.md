# Word Generation

[Back to Wordplay](../README.md) | [Try it live](https://djdhillxn.github.io/projects/hangman/) | [Open the simulator notebook](../notebooks/gameSimulator.ipynb)

LexiNet can be used as a generative model because its count tables define probability distributions over letters conditioned on surrounding characters. The result is not dictionary retrieval and it is not a neural text generator. It is a bidirectional sampler that constructs a fixed-length string from the same local evidence used by the Hangman player.

## Generating from a blank board

Generation starts with a requested length, such as nine characters:

```text
_ _ _ _ _ _ _ _ _
```

For every remaining blank position, the generator evaluates every letter `a` through `z`. It obtains a forward probability from the characters to the left and a reverse probability from the characters to the right, then multiplies them:

$$
s(i,x)=P_f(x\mid L_i)P_b(x\mid R_i).
$$

This produces a distribution over `(position, letter)` pairs rather than only over the next letter. One pair is sampled, its character is written into the board, and the process repeats with the newly available context.

```text
_ _ _ _ _ _ _ _ _
_ _ _ _ _ _ i _ _
_ _ _ _ _ _ i _ n
_ _ _ _ a _ i _ n
...
```

The model can therefore choose an informative interior or final character before filling the beginning of the word. That is the main difference from ordinary left-to-right generation.

## Temperature and model order

The sampler converts scores into weights relative to the best available score. Temperature controls how concentrated the resulting choice is. A low value repeatedly favors the strongest patterns and tends to produce conservative strings; a higher value gives weaker candidates more opportunity and produces less familiar combinations. A temperature of zero selects the maximum-scoring pair directly.

By default, generation uses the same word-length mapping as the game simulator: trigrams for lengths 1-2, 4-grams for length 3, and 6-grams for longer words when all local models are loaded. The browser deployment can use a smaller maximum order to keep its JSON asset practical. If no contextual candidate has a positive score, the notebook falls back to the learned unigram character counts.

The complete notebook implementation appears in the **Generate New Model Words** section of [`notebooks/gameSimulator.ipynb`](../notebooks/gameSimulator.ipynb). [`src/export_lexinet_web_model.py`](../src/export_lexinet_web_model.py) converts the trained pickle files into compact JSON tables for the portfolio deployment.

## What the output means

A generated string is evidence that the model found a sequence compatible with character patterns in its training corpus. It is not guaranteed to be absent from the dictionary, pronounceable, meaningful, or culturally unused. The generator currently does not perform a novelty check against the train and test lists.

That makes the output useful as a prompt rather than a declaration. It can suggest names for fictional places, products, characters, or playful concepts; before adopting a generated word, check its existing meanings, associations, and usage. The live page supports exactly this kind of language play: generate a candidate, decide what it ought to mean, and see whether the word deserves a life beyond its probability table.

## Generation and guessing

The generator and the Hangman player share the same probabilities but make different decisions. The player sums a letter's evidence across every blank and greedily guesses the best letter, because one guess reveals all of its occurrences. The generator instead samples one `(position, letter)` pair and permanently fills only that slot. One system is optimizing a six-life game; the other is exploring the distribution learned by the model.
