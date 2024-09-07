# Blackjack AI using PyTorch DQN

## Overview

This project features a Blackjack-playing AI developed using a Deep Q-Network (DQN) built with PyTorch. My best model achieves a 47.6% win rate and 6.1% push rate evaluated over 1000 games.

## Prerequisites

To run this project, you will need:

- Python 3.x (tested on 3.12) or Docker

## Get started

### Using Docker (recommended)

To use Docker, simply install the repository, build the image, and run the container:

```bash
git clone https://github.com/SamuelReeder/blackjack-ai.git
cd blackjack-ai
make all
```

### Manually using pip

Clone the repository and install the required packages:

```bash
git clone https://github.com/SamuelReeder/blackjack-ai.git
cd blackjack-ai
pip install -r requirements.txt
```

## Training the Model

To train a new model, use the following command, replacing <model_num> with the desired model number:

```bash
python src/run.py <model_num>
```

## Evaluating the Model

To evaluate the performance of a trained model, use the same model number with the evaluation script:

```bash
python src/eval.py <model_num>
```

Use `python src/eval.py 21` for my best pretrained model!
