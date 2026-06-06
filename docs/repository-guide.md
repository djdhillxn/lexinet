# Wordplay Repository Guide

[Back to Wordplay](../README.md) | [Word generation](word-generation.md) | [Live demo](https://djdhillxn.github.io/projects/hangman/)

Wordplay separates raw word lists, trained count dictionaries, reusable Python modules, and exploratory notebooks. The project name is Wordplay; **LexiNet** remains the name used for the trained model and its exported artifacts.

```text
.
|-- README.md
|-- data
|   |-- train/words_train.txt
|   `-- test/words_test.txt
|-- docs
|   |-- repository-guide.md
|   `-- word-generation.md
|-- game_results.csv
|-- notebooks
|   |-- documentation.md
|   |-- evaluation.ipynb
|   |-- gameSimulator.ipynb
|   `-- training_and_eda.ipynb
|-- requirements.txt
|-- results
|   `-- models
|       |-- n_3_gram_model_kneser_ney.pkl
|       |-- n_4_gram_model_kneser_ney.pkl
|       |-- n_5_gram_model_kneser_ney.pkl
|       |-- n_6_gram_model_kneser_ney.pkl
|       `-- lexinet_web_model_3_6.json
`-- src
    |-- data_preparation.py
    |-- evaluate.py
    |-- export_lexinet_web_model.py
    |-- game_simulator.py
    |-- player_agent.py
    `-- train.py
```

## Data and notebooks

[`data/train/words_train.txt`](../data/train/words_train.txt) contains 227,300 lowercase, unique training words. [`data/test/words_test.txt`](../data/test/words_test.txt) contains 170,671 lowercase, unique held-out words, with no overlap between the two files.

[`notebooks/training_and_eda.ipynb`](../notebooks/training_and_eda.ipynb) is the main training narrative. It loads the corpus, trains forward and reverse models for orders 3-6 through `src.train.NgramModel`, and inspects the distribution of word lengths and its sparse extremes.

[`notebooks/evaluation.ipynb`](../notebooks/evaluation.ipynb) is the reproducible evaluation surface. It reports forward perplexity on both splits and runs the complete six-life simulator, including grouped results by word length. The notebook exposes worker and multiprocessing settings so large simulations can be moved to an environment such as Colab.

[`notebooks/gameSimulator.ipynb`](../notebooks/gameSimulator.ipynb) is the interactive surface. It loads trained model files, animates a game against a chosen secret word, supports small batch trials, and contains the bidirectional word generator described in [Word generation](word-generation.md).

[`notebooks/documentation.md`](../notebooks/documentation.md) is an API-oriented description of the simulator class and remains useful when the concern is method signatures rather than the modeling argument.

## Python modules

| Module | Responsibility |
|---|---|
| [`src/data_preparation.py`](../src/data_preparation.py) | Loads line-delimited word lists and checks lowercase input. |
| [`src/train.py`](../src/train.py) | Builds exact and masked forward/reverse count tables and serializes model bundles. |
| [`src/player_agent.py`](../src/player_agent.py) | Converts counts to probabilities, interpolates orders, combines directions, and selects guesses. |
| [`src/game_simulator.py`](../src/game_simulator.py) | Runs one game or a full corpus simulation and groups outcomes by word length. |
| [`src/evaluate.py`](../src/evaluate.py) | Calculates forward perplexity for saved models and optionally writes a CSV report. |
| [`src/export_lexinet_web_model.py`](../src/export_lexinet_web_model.py) | Converts pickle model bundles into compact browser-readable JSON. |

The serialized bundles contain forward n-grams, reverse n-grams, unigram counts, and continuation-count structures used by the smoothing experiments. The default `best` player currently scores interpolated conditional counts with add-\(k\) smoothing; the Kneser-Ney methods remain available in `player_agent.py` for comparison.

## Training and evaluation

From the repository root, the script entry point trains the order selected near the bottom of `src/train.py`:

```bash
python3 -m src.train
```

The training notebook is the convenient route for producing all four orders in one run. Saved models are written beneath `results/models/`.

The evaluation module calculates perplexity for orders 3-6 on both data splits:

```bash
python3 -m src.evaluate
```

It can persist the report without changing the notebook:

```bash
python3 -m src.evaluate --output-csv results/perplexity.csv
```

Full game simulations are intentionally notebook-driven because they are substantially more expensive than perplexity evaluation. The worker count, start method, model run name, and optional sample size are all visible near the top of the simulation section in `notebooks/evaluation.ipynb`.

## Browser export

The web exporter retains the forward tables, reverse tables, and unigram counts needed by the JavaScript demo, converts `<s>` and `</s>` to compact symbols, and writes deterministic JSON:

```bash
python3 -m src.export_lexinet_web_model \
  --models-dir results/models \
  --orders 3 4 5 6 \
  --out results/models/lexinet_web_model_3_6.json \
  --gzip-copy
```

The live portfolio may choose fewer orders because the 6-gram table materially increases download size. This repository keeps the fuller export for local inspection and deployment experiments.
