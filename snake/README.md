# Snake Reinforcement Learning Project
Angela Bortolato, ID 2156562, Physics of Data.

Project developed for Reinforcement Learning coursework.

---

## Project Overview

This project investigates reinforcement learning for the Snake game under two observation settings:
- Fully observable environment
- Partially observable environment

Two agents are implemented and compared:
- Deep Q-Network (DQN)
- Proximal Policy Optimization (PPO)

The goal is to analyze performance differences across algorithms and observation constraints.

---

## Project Structure
```text
snake/
├── models/
│ ├── dqn_snake.keras
│ ├── dqn_snake_partial.keras
│ ├── ppo_actor_snake.keras
│ └── ppo_actor_snake_partial.keras
│
├── results/
│ ├── dqn_training_metrics.csv
│ ├── dqn_training_metrics_partial.csv
│ ├── ppo_training_metrics.csv
│ └── ppo_training_metrics_partial.csv
│
├── __init__.py
├── baselines.py
├── environments_fully_observable.py
├── environments_partially_observable.py
├── evaluate_fn.py
├── training_fn.py
├── run_evaluate.py
├── run_training.py
├── main.ipynb
└── README.md
```

---

## Requirements

Install the required Python packages before running the project.
```bash
pip install -r requirements.txt
```

---

## Running the Project

### Evaluate pretrained models
To evaluate the pretrained agents:
```bash
python run_evaluate.py
```

This script:
* loads the saved trained models from the models/ folder
* evaluates the agents
* computes performance metrics
* optionally displays animations and plots

### Train the agents
To retrain the models:
```bash
python run_training.py
```

This script:
* trains the RL agents
* saves trained models inside models/
* saves training metrics inside results/

---

## Notebook
The notebook main.ipynb contains:
* development experiments
* intermediate analyses
* interactive visualizations/widgets
* exploratory tests

The notebook is not required to run the project evaluation.

---

## Notes

* All file paths are handled relatively, so the project should run correctly after extracting the ZIP archive.
* Pretrained .keras models are already included in the repository.
* Generated CSV metric files are already included in the results/ folder.