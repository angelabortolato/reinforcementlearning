from baselines import greedy_policy, zigzag_policy, semi_blind_policy
from evaluate_fn import run_evaluation, plot_RL_vs_baselines
from training_fn import dqn_policy, ppo_policy, get_env

import numpy as np
import pandas as pd

import random
import tensorflow as tf

def reset_seed():
    tf.random.set_seed(0)
    random.seed(0)
    np.random.seed(0)

# Global Hyperparameters & Environment Parameter Framework
CONFIG = {

    "N_BOARDS_TRAIN": 128,      # Vectorized batch processing size for training
    "N_BOARDS_EVAL": 1000,     # High-volume batch for stable baseline/evaluation averages
    "BOARD_SIZE": 8,           # Must be matching everywhere (8x8)
    "MASK SIZE": 2,            # For partially observable envs, defines local view radius
    
    "GAMMA": 0.9,              # Common discount factor parameter
    "BUFFER_CAPACITY": 100000,   # Replay buffer size for experience storage
    "BATCH_SIZE": 500,          # Memory sampling mini-batch
    "TRAINING_STEPS": 30000,        # Total DQN optimization timeline steps
    
    "EPSILON_START": 1.0,
    "EPSILON_END": 0.15,
    "EPSILON_DECAY": 0.9999,
    "EVAL_FREQ": 500,
    
    "TARGET_UPDATE_FREQ": 20,
    "LEARNING_RATE": 0.001,
    "TAU" : 0.005,          # Soft update blend factor
    
    # Specific PPO parameters
    "ROLLOUT_STEPS": 20,      # Environmental timeline steps captured per epoch loop execution
    "PPO_EPOCHS": 4,          # Updates per evaluation pass batch
    "CLIP_EPSILON": 0.2       # Stability bounds window clipping
}

#########################################################
#                 FULLY OBSERVABLE
#########################################################
print("Evaluating Fully Observable Environment...\n")
#------------ BASELINES EVALUATION --------------
reset_seed()
env = get_env(CONFIG)
greedy_stats=run_evaluation(env, greedy_policy, "Greedy Baseline", max_steps=500, gamma=CONFIG["GAMMA"], verbose=True)

reset_seed()
env = get_env(CONFIG)
zigzag_stats=run_evaluation(env, zigzag_policy, "ZigZag Baseline", max_steps=500, gamma=CONFIG["GAMMA"], verbose=True)

#------------ DQN EVALUATION --------------
# Load the saved training data
try:
    loaded_dqn_stats = pd.read_csv("results/dqn_training_metrics.csv")
    print("Successfully loaded saved DQN metrics!")

except FileNotFoundError:
    print("Error: Could not find your saved CSV file. Make sure you've run the training and saved it first!")

# load trained model and weights
q_network = tf.keras.models.load_model("models/dqn_snake.keras")

# evaluate
dqn_policy_fn = lambda e: dqn_policy(e, q_network)
reset_seed()
env = get_env(CONFIG)
dqn_stats_final=run_evaluation(env, dqn_policy_fn, "DQN Agent", max_steps=500, gamma=CONFIG["GAMMA"], verbose=True)

#------------ PPO EVALUATION --------------
# Load the saved training data
try:
    loaded_ppo_stats = pd.read_csv("results/ppo_training_metrics.csv")
    print("Successfully loaded saved PPO metrics!")

except FileNotFoundError:
    print("Error: Could not find your saved CSV file. Make sure you've run the training and saved it first!")

ppo_actor = tf.keras.models.load_model("models/ppo_actor_snake.keras")

# evaluate
ppo_policy_fn = lambda e: ppo_policy(e, ppo_actor)
reset_seed()
env = get_env(CONFIG)
ppo_stats_final=run_evaluation(env, ppo_policy_fn, "PPO Agent", max_steps=500, gamma=CONFIG["GAMMA"], verbose=True)


#########################################################
#                 PARTIALLY OBSERVABLE
#########################################################
print("Evaluating Partially Observable Environment...\n")

#------------ BASELINE EVALUATION --------------
reset_seed()
env = get_env(CONFIG, full=False)
semi_blind_stats=run_evaluation(env, semi_blind_policy, "Semi-Blind Baseline", max_steps=100, gamma=CONFIG["GAMMA"], verbose=True)

#------------ DQN EVALUATION --------------
# Load the saved training data
try:
    loaded_dqn_stats_partial = pd.read_csv("results/dqn_training_metrics_partial.csv")
    print("Successfully loaded saved DQN metrics!")

except FileNotFoundError:
    print("Error: Could not find your saved CSV file. Make sure you've run the training and saved it first!")

# load trained model and weights
q_network = tf.keras.models.load_model("models/dqn_snake_partial.keras")

# evaluate
dqn_policy_fn = lambda e: dqn_policy(e, q_network)
reset_seed()
env = get_env(CONFIG, full=False)
dqn_stats_final=run_evaluation(env, dqn_policy_fn, "DQN Agent", max_steps=500, gamma=CONFIG["GAMMA"], verbose=True)

#------------ PPO EVALUATION --------------
# Load the saved training data
try:
    loaded_ppo_stats_partial = pd.read_csv("results/ppo_training_metrics_partial.csv")
    print("Successfully loaded saved PPO metrics!")

except FileNotFoundError:
    print("Error: Could not find your saved CSV file. Make sure you've run the training and saved it first!")

ppo_actor = tf.keras.models.load_model("models/ppo_actor_snake_partial.keras")

# evaluate
ppo_policy_fn = lambda e: ppo_policy(e, ppo_actor)
reset_seed()
env = get_env(CONFIG, full=False)
ppo_stats_final=run_evaluation(env, ppo_policy_fn, "PPO Agent", max_steps=500, gamma=CONFIG["GAMMA"], verbose=True)

