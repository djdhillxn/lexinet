# LexiNet

LexiNet is a project focused on building and evaluating n-gram language models for word prediction and guessing games. This repository includes code for training models, simulating games, and evaluating model performance.

## Results
Training on [train set](/data/train/words_train.txt) \
Validatiaon on [val set](/data/test/words_test.txt)


[Validation Set Results](game_results.csv) \
Total Games: 170671 \
Games Won: 109689 \
Accuracy: 64.27 %

## Repository Structure

```
~/Garage/lexinet $ tree -L 2
.
├── README.md
├── data
│   ├── test
│   └── train
├── game_results.csv
├── notebooks
│   └── EDA.ipynb
├── perplexity.md
├── requirements.txt
├── results
│   └── models
└── src
    ├── __init__.py
    ├── data_preparation.py
    ├── documentation.md
    ├── evaluate.py
    ├── game_simulator.py
    ├── player_agent.py
    └── train.py
```

## N-Gram Language Models

Find training and game simulator documentation [here](src/documentation.md) 