# Blackjack AI using PyTorch DQN

## Overview
This project features a Blackjack-playing AI developed using a Deep Q-Network (DQN) built with PyTorch. The AI has been rigorously trained and evaluated on a diverse set of 50,000 games, achieving an impressive win rate of 52.75%.

## Prerequisites
To run this project, you will need:
- Python 3.8 or later
- PyTorch
- Other dependencies listed in `requirements.txt`

## Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/SamuelReeder/blackjack-ai.git
cd blackjack-ai
pip install -r requirements.txt
```

## Training the Model
To train a new model, use the following command, replacing <model_num> with the desired model number:
```
py run.py <model_num>
```

## Evaluating the Model
To evaluate the performance of a trained model, use the same model number with the evaluation script:
```
py eval.py <model_num>
```

## Run one a gambling site
Use the browser extension here at this [repo](https://github.com/SamuelReeder/web-action-encoder) to make a JSON file encoding various actions for the specific site and then pass it in the game_manager.py file. Use the reference for [247blackjack.com] in the web-metadata folder. 

Then run:
```
py src/oneline_play.py <model_num>
```

## Results
The best current model achieves a win rate of 52.75% over an evaluation set of 50,000 games.

## Contributing
Contributions to this project are welcome. Please open an issue or submit a pull request.
