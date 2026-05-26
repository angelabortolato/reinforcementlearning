from training_fn import train_dqn, train_ppo

import numpy as np
import os

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

#------------- DQN ---------------
dqn_stats= train_dqn(CONFIG)
os.makedirs("results", exist_ok=True)

dqn_stats.to_csv("results/dqn_training_metrics.csv", index=False)
print("Evaluation metrics successfully saved to 'results/dqn_training_metrics.csv'!")

#------------- PPO ---------------
ppo_stats = train_ppo(CONFIG)

os.makedirs("results", exist_ok=True)

ppo_stats.to_csv("results/ppo_training_metrics.csv", index=False)
print("Evaluation metrics successfully saved to 'results/ppo_training_metrics.csv'!")

#########################################################
#                 PARTIALLY OBSERVABLE
#########################################################

#------------- DQN ---------------
dqn_stats_partial= train_dqn(CONFIG, full=False)
os.makedirs("results", exist_ok=True)

dqn_stats_partial.to_csv("results/dqn_training_metrics_partial.csv", index=False)
print("Evaluation metrics successfully saved to 'results/dqn_training_metrics_partial.csv'!")

#------------- PPO ---------------
ppo_stats_partial = train_ppo(CONFIG, full=False)

os.makedirs("results", exist_ok=True)

ppo_stats_partial.to_csv("results/ppo_training_metrics_partial.csv", index=False)
print("Evaluation metrics successfully saved to 'results/ppo_training_metrics_partial.csv'!")
