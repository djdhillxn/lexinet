# src/evaluate.py

import argparse
import csv
import math
import pickle
from pathlib import Path

import nltk
from nltk.util import ngrams
from collections import defaultdict, Counter
import os
from tqdm import tqdm

try:
    from src.data_preparation import load_data, preprocess_data
except ImportError:
    from data_preparation import load_data, preprocess_data


DEFAULT_N_VALUES = (3, 4, 5, 6)
DEFAULT_MODEL_TEMPLATE = "n_{n}_gram_model_kneser_ney.pkl"
DEFAULT_FLOOR_VALUE = 1e-12

def iter_ngrams(tokens, n):
    for index in range(len(tokens) - n + 1):
        yield tuple(tokens[index:index + n])


def load_words(data_path):
    return preprocess_data(load_data(str(data_path)))


class NgramModel:
    """Forward n-gram model evaluator for the dictionaries saved by train.py."""

    def __init__(self, n, model_data=None):
        self.n = n
        self.model_data = None
        self.ngrams = None
        if model_data is not None:
            self.set_model_data(model_data)

    def set_model_data(self, model_data):
        self.model_data = model_data
        if isinstance(model_data, dict) and "ngrams" in model_data:
            self.ngrams = model_data["ngrams"]
        else:
            self.ngrams = model_data
        if self.ngrams is None:
            raise ValueError("Model data does not contain forward n-gram counts.")
        return self

    def load(self, file_path):
        with open(file_path, "rb") as file:
            return self.set_model_data(pickle.load(file))

    def load_trained_models_dict(self, models_dir, model_name):
        model_path = Path(models_dir) / model_name
        return self.load(model_path)

    def conditional_probability(self, prefix, suffix):
        prefix_counts = self.ngrams.get(prefix)
        if not prefix_counts:
            return 0.0
        prefix_total = sum(prefix_counts.values())
        if prefix_total <= 0:
            return 0.0
        return prefix_counts.get(suffix, 0) / prefix_total

    def perplexity_stats(self, data, floor_value=DEFAULT_FLOOR_VALUE, include_end_tokens=False):
        """Calculate simple forward perplexity over predicted characters.

        train.py skips n-grams whose target suffix is </s>, so this evaluator
        skips </s> targets by default. That keeps the scored event space aligned
        with the events the saved forward model actually learned.
        """
        negative_log_likelihood = 0.0
        token_count = 0
        zero_probability_count = 0

        for word in data:
            padded_word = ["<s>"] * (self.n - 1) + list(word) + ["</s>"] * (self.n - 1)
            for gram in iter_ngrams(padded_word, self.n):
                prefix = gram[:-1]
                suffix = gram[-1]
                if not include_end_tokens and suffix == "</s>":
                    continue

                probability = self.conditional_probability(prefix, suffix)
                if probability <= 0.0:
                    zero_probability_count += 1
                    if floor_value is None:
                        return {
                            "perplexity": math.inf,
                            "average_negative_log_likelihood": math.inf,
                            "token_count": token_count + 1,
                            "zero_probability_count": zero_probability_count,
                            "floor_value": None,
                        }
                    probability = floor_value

                negative_log_likelihood -= math.log(probability)
                token_count += 1

        if token_count == 0:
            raise ValueError("No tokens were available for perplexity evaluation.")

        average_negative_log_likelihood = negative_log_likelihood / token_count
        return {
            "perplexity": math.exp(average_negative_log_likelihood),
            "average_negative_log_likelihood": average_negative_log_likelihood,
            "token_count": token_count,
            "zero_probability_count": zero_probability_count,
            "floor_value": floor_value,
        }

    def perplexity(self, data, floor_value=DEFAULT_FLOOR_VALUE, include_end_tokens=False):
        return self.perplexity_stats(
            data,
            floor_value=floor_value,
            include_end_tokens=include_end_tokens,
        )["perplexity"]


def model_path_for_n(models_dir, n, model_file_template=DEFAULT_MODEL_TEMPLATE):
    return Path(models_dir) / model_file_template.format(n=n)


def evaluate_model_set(
    models_dir,
    train_data_path,
    test_data_path,
    n_values=DEFAULT_N_VALUES,
    model_file_template=DEFAULT_MODEL_TEMPLATE,
    floor_value=DEFAULT_FLOOR_VALUE,
    include_end_tokens=False,
):
    datasets = {
        "train": load_words(train_data_path),
        "test": load_words(test_data_path),
    }
    rows = []

    for n in n_values:
        model_path = model_path_for_n(models_dir, n, model_file_template)
        model = NgramModel(n)
        model.load(model_path)
        for split, words in datasets.items():
            stats = model.perplexity_stats(
                words,
                floor_value=floor_value,
                include_end_tokens=include_end_tokens,
            )
            rows.append({
                "n": n,
                "split": split,
                "model_file": str(model_path),
                **stats,
            })

    return rows


def format_perplexity(value):
    if math.isinf(value):
        return "inf"
    if value >= 1_000_000:
        return f"{value:.6e}"
    return f"{value:.6f}"


def print_results_table(rows):
    columns = [
        "n",
        "split",
        "perplexity",
        "token_count",
        "zero_probability_count",
        "floor_value",
    ]
    printable_rows = []
    for row in rows:
        printable_rows.append({
            "n": row["n"],
            "split": row["split"],
            "perplexity": format_perplexity(row["perplexity"]),
            "token_count": row["token_count"],
            "zero_probability_count": row["zero_probability_count"],
            "floor_value": row["floor_value"],
        })

    widths = {
        column: max(len(column), *(len(str(row[column])) for row in printable_rows))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    divider = "-+-".join("-" * widths[column] for column in columns)
    print(header)
    print(divider)
    for row in printable_rows:
        print(" | ".join(str(row[column]).ljust(widths[column]) for column in columns))


def write_results_csv(rows, output_csv_path):
    columns = [
        "n",
        "split",
        "perplexity",
        "average_negative_log_likelihood",
        "token_count",
        "zero_probability_count",
        "floor_value",
        "model_file",
    ]
    with open(output_csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate simple forward n-gram perplexity.")
    parser.add_argument("--models-dir", default="results/models")
    parser.add_argument("--model-template", default=DEFAULT_MODEL_TEMPLATE)
    parser.add_argument("--train-data", default="data/train/words_train.txt")
    parser.add_argument("--test-data", default="data/test/words_test.txt")
    parser.add_argument("--n-values", nargs="+", type=int, default=list(DEFAULT_N_VALUES))
    parser.add_argument("--floor-value", type=float, default=DEFAULT_FLOOR_VALUE)
    parser.add_argument(
        "--strict-zero-probability",
        action="store_true",
        help="Return infinite perplexity when an evaluated event has zero probability.",
    )
    parser.add_argument(
        "--include-end-tokens",
        action="store_true",
        help="Also score </s> targets. Disabled by default because train.py skips them.",
    )
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    floor_value = None if args.strict_zero_probability else args.floor_value
    rows = evaluate_model_set(
        models_dir=args.models_dir,
        train_data_path=args.train_data,
        test_data_path=args.test_data,
        n_values=args.n_values,
        model_file_template=args.model_template,
        floor_value=floor_value,
        include_end_tokens=args.include_end_tokens,
    )
    print_results_table(rows)
    if args.output_csv:
        write_results_csv(rows, args.output_csv)


if __name__ == "__main__":
    main()


"""
# src/evaluate.py

class NgramModel:
    def __init__(self, n):
        self.n = n
        self.ngrams = None

    def load(self, file_path):
        with open(file_path, 'rb') as file:
            self.ngrams = pickle.load(file)

    def perplexity(self, data):
        total_log_prob = 0
        total_length = 0
        floor_value = 1e-6  # Updated floor value

        for word in data:
            padded_word = ['<s>'] * (self.n - 1) + list(word) + ['</s>']
            for gram in ngrams(padded_word, self.n):
                prefix = gram[:-1]
                suffix = gram[-1]
                count_prefix = sum(self.ngrams[prefix].values())
                count_suffix = self.ngrams[prefix][suffix]
                probability = count_suffix / count_prefix if count_prefix > 0 else floor_value

                if probability <= 0:
                    #print(f"Debug: Non-positive probability encountered. prefix={prefix}, suffix={suffix}, count_prefix={count_prefix}, count_suffix={count_suffix}")
                    probability = floor_value

                total_log_prob += -math.log2(probability)
                total_length += 1

        return math.pow(2, total_log_prob / total_length)

# Load and preprocess test data
test_data_path = "../data/test/words_test.txt"
test_data = preprocess_data(load_data(test_data_path))
model_dir = "../results/models"

# Evaluate n-gram models for n=2 to 7
for n in range(2, 8):
    model_path = os.path.join(model_dir, f"n_{n}_gram_model_corrected.pkl")
    ngram_model = NgramModel(n)
    ngram_model.load(model_path)
    perplexity = ngram_model.perplexity(test_data)
    print(f"Perplexity for {n}-gram model: {perplexity}")
"""