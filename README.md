# Wordplay

### Bidirectional character n-gram models for guessing and generating words

[Try the live model](https://djdhillxn.github.io/projects/hangman/) | [Training and EDA](notebooks/training_and_eda.ipynb) | [Evaluation](notebooks/evaluation.ipynb) | [Word generation](docs/word-generation.md) | [Repository guide](docs/repository-guide.md)

Wordplay studies a small but unusual language-modeling problem: can a model recover a hidden word when it sees only its length, a few revealed characters, and the letters that have already failed? The model behind the project is called **LexiNet**. It is trained on character n-grams of orders 3 through 6, reads evidence from both sides of every blank, and plays a Hangman-style game with six incorrect guesses available.

The [live browser demo](https://djdhillxn.github.io/projects/hangman/) lets LexiNet attempt any word you enter. The same page can also sample new word-like strings from the learned distributions; that generative procedure is explained separately in [Word generation](docs/word-generation.md).

## From language modeling to Wordplay

Consider a conventional next-token prompt:

> I liked it. I was good at --

A trained language model, especially one familiar with *Breaking Bad*, would assign high probability to `it`. It estimates a continuation from the context on the left. Hangman is less cooperative. A state such as `c_l_mb_a` contains several missing characters, and each blank has useful context on both sides.

N-grams can be defined over words, subwords, or characters. Wordplay uses characters: a 6-gram represents five context symbols and one character to predict. The training code learns the usual forward conditional distribution

$$
P_f(x \mid l_1,\ldots,l_{n-1})
$$

and a reverse distribution

$$
P_b(x \mid r_1,\ldots,r_{n-1}),
$$

where the second model asks which character is likely to appear immediately before a known right context. The implementation lives in [`src/train.py`](src/train.py#L43-L103), while the game-time scorer lives in [`src/player_agent.py`](src/player_agent.py#L14-L107).

## A three-word training corpus

The shortest way to understand the stored model is to train a trigram on the toy corpus `cat`, `car`, and `can`. For `n = 3`, each word receives two start symbols and two end symbols. The resulting windows are:

```text
cat  ->  <s><s>c   <s>ca   cat   at</s>   t</s></s>
car  ->  <s><s>c   <s>ca   car   ar</s>   r</s></s>
can  ->  <s><s>c   <s>ca   can   an</s>   n</s></s>
```

The forward model treats the first two symbols as context and the third as the prediction. It does not learn `</s>` as a target, so the final two windows of each row are omitted from the forward table. The reverse model treats the final two symbols as context and predicts the symbol before them; windows beginning with `<s>` are omitted there.

Wordplay also stores masked contexts. The idea is inspired by BERT's masked-language-model objective, but the mechanism is different: no neural representation is learned and no character is masked at random during optimization. Instead, the trainer explicitly replaces eligible context characters with `_` and increments those count tables too. A future board such as `c_` can therefore query a context the model saw during training.

For this toy corpus, the complete forward store is:

| Left context | Next-character counts |
|---|---|
| `(<s>, <s>)` | `{c: 3}` |
| `(<s>, c)` | `{a: 3}` |
| `(<s>, _)` | `{a: 3}` |
| `(c, a)` | `{t: 1, r: 1, n: 1}` |
| `(_, a)` | `{t: 1, r: 1, n: 1}` |
| `(c, _)` | `{t: 1, r: 1, n: 1}` |

The reverse store contains the complementary evidence:

| Right context | Previous-character counts |
|---|---|
| `(a, t)`, `(a, r)`, `(a, n)` | `{c: 1}` for each context |
| `(_, t)`, `(_, r)`, `(_, n)` | `{c: 1}` for each context |
| `(a, _)` | `{c: 3}` |
| `(t, </s>)`, `(r, </s>)`, `(n, </s>)` | `{a: 1}` for each context |
| `(_, </s>)` | `{a: 3}` |
| `(</s>, </s>)` | `{t: 1, r: 1, n: 1}` |

This is the whole training idea at corpus scale: build exact and masked count dictionaries for every order from 3 through 6, in both directions. The [training and EDA notebook](notebooks/training_and_eda.ipynb) runs that process over 227,300 unique training words and also examines the corpus by word length. The training and held-out sets are disjoint, both have median length 9, and roughly 68% of training words and 71% of test words fall between lengths 7 and 12.

## From counts to a guess

For an observed context, the current player turns counts into an add-\(k\) estimate with \(k = 0.05\):

$$
P_f(x \mid L)=
\frac{C(L,x)+k}
{\sum_{a \in \mathcal{A}} C(L,a)+26k}.
$$

The reverse probability is calculated in the same way. For words of length 9 or less, the estimate is recursively interpolated down to the trigram model:

$$
\widehat{P}^{(n)}
=0.8P^{(n)}+0.2\widehat{P}^{(n-1)}.
$$

Starting at the 6-gram, this gives effective weights of `0.8`, `0.16`, `0.032`, and `0.008` to orders 6, 5, 4, and 3. This is interpolation rather than hard backoff: lower-order evidence contributes even when the higher-order context exists.

For words longer than 9 characters, the current benchmark configuration uses the highest selected order without interpolation and assigns `1/26` to an unseen context. This cutoff is an empirical game heuristic, not a general property of language models. Longer boards provide more blank positions and more opportunities for a specific high-order pattern to contribute; in the recorded experiments, retaining that sharper evidence worked better than blending in shorter contexts.

At each turn, the player evaluates every unguessed letter at every blank. It does not enumerate complete word fillings. For candidate letter \(x\), it multiplies the forward and reverse estimates at blank \(i\), treating the two directional signals as a practical conditional-independence approximation, and then sums the evidence across the board:

$$
\operatorname{score}(x)
=\sum_{i \in \text{blanks}}
P_f(x \mid L_i)\,P_b(x \mid R_i).
$$

The highest-scoring letter is guessed, and every occurrence of that letter is revealed. On `co_umb_a`, for example, the left context favors a continuation after `co`, while the right context `umb_a` strongly constrains what can precede it. Their product makes `l` a compelling candidate; after that guess reveals `columb_a`, the same calculation can favor `i`.

## Evaluation

The [evaluation notebook](notebooks/evaluation.ipynb) provides two views of model quality. Intrinsic evaluation uses forward perplexity,

$$
\operatorname{PPL}
=\exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(x_i\mid x_{<i})\right),
$$

with a configurable probability floor for unseen events. Lower perplexity means the model assigned greater average likelihood to the observed characters, but the value depends on the floor and evaluates only the forward model. The repository therefore treats the six-life game simulation as the more meaningful extrinsic result.

The table combines the recorded training simulation with the committed held-out results in [`game_results.csv`](game_results.csv). Each cell reports `wins / games (win rate)`.

| Length | Training split | Held-out split |
|---:|---:|---:|
| 1 | 4/17 (23.53%) | 1/2 (50.00%) |
| 2 | 21/264 (7.95%) | 10/41 (24.39%) |
| 3 | 175/2,201 (7.95%) | 60/531 (11.30%) |
| 4 | 850/5,287 (16.08%) | 378/2,580 (14.65%) |
| 5 | 2,806/11,274 (24.89%) | 1,268/6,456 (19.64%) |
| 6 | 6,639/19,541 (33.97%) | 3,757/13,112 (28.65%) |
| 7 | 11,576/25,948 (44.61%) | 7,507/19,156 (39.19%) |
| 8 | 17,147/30,452 (56.31%) | 12,554/24,119 (52.05%) |
| 9 | 21,243/30,906 (68.73%) | 15,700/24,495 (64.09%) |
| 10 | 21,008/26,953 (77.94%) | 15,879/21,657 (73.32%) |
| 11 | 19,568/22,786 (85.88%) | 14,677/17,823 (82.35%) |
| 12 | 16,558/18,178 (91.09%) | 12,178/13,769 (88.45%) |
| 13 | 12,224/12,956 (94.35%) | 9,448/10,118 (93.38%) |
| 14 | 8,415/8,710 (96.61%) | 6,587/6,896 (95.52%) |
| 15 | 5,108/5,211 (98.02%) | 4,322/4,454 (97.04%) |
| 16 | 3,096/3,143 (98.50%) | 2,484/2,534 (98.03%) |
| 17 | 1,751/1,775 (98.65%) | 1,449/1,474 (98.30%) |
| 18 | 855/859 (99.53%) | 728/743 (97.98%) |
| 19 | 435/441 (98.64%) | 386/390 (98.97%) |
| 20 | 224/225 (99.56%) | 169/171 (98.83%) |
| 21 | 98/98 (100.00%) | 87/88 (98.86%) |
| 22 | 44/44 (100.00%) | 32/34 (94.12%) |
| 23 | 14/14 (100.00%) | 16/16 (100.00%) |
| 24 | 9/9 (100.00%) | 3/3 (100.00%) |
| 25 | 3/3 (100.00%) | 4/4 (100.00%) |
| 27 | 2/2 (100.00%) | 1/1 (100.00%) |
| 28 | 1/1 (100.00%) | 2/2 (100.00%) |
| 29 | 2/2 (100.00%) | n/a |
| 31 | n/a | 1/1 (100.00%) |
| 45 | n/a | 1/1 (100.00%) |
| **All lengths** | **149,876/227,300 (65.94%)** | **109,689/170,671 (64.27%)** |

The rise in win rate with length is real but should be read alongside sample size: the test set contains tens of thousands of words around lengths 8-11 and only a handful beyond length 23. Long words expose more character positions to the scorer, while very short words provide little context before the six-life budget is exhausted. Runs can differ by a few games because the agent uses a random choice when all candidate scores are empty.

## Working with the repository

The notebook path is the clearest tour. [`training_and_eda.ipynb`](notebooks/training_and_eda.ipynb) trains orders 3-6 and explores the length distribution; [`evaluation.ipynb`](notebooks/evaluation.ipynb) calculates perplexity and runs train/test game simulations; and [`gameSimulator.ipynb`](notebooks/gameSimulator.ipynb) provides an interactive game plus word generation.

The same operations are available as Python modules. From the repository root, the following commands train the default order configured in `src/train.py` and evaluate the saved orders 3-6:

```bash
python3 -m src.train
python3 -m src.evaluate
```

The browser model is produced by [`src/export_lexinet_web_model.py`](src/export_lexinet_web_model.py), which compacts the pickle dictionaries into JSON-safe forward, reverse, and unigram tables. See the [repository guide](docs/repository-guide.md) for the full layout, model artifacts, and common workflows.

## References

The language-modeling notation and treatment of n-grams, interpolation, perplexity, and Kneser-Ney smoothing follow Dan Jurafsky and James H. Martin's [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/), especially Chapter 3 and Appendix C.

The masked-context idea was motivated by Devlin et al.'s [BERT paper](https://arxiv.org/abs/1810.04805). Wordplay borrows the intuition that missing symbols can be predicted from context, while implementing it as explicit character-count tables tailored to partially revealed words.
