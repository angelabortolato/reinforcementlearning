import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
import random
from collections import deque
import environments_fully_observable 
import environments_partially_observable 
from evaluate import run_evaluation
import os

def get_env(config, n=None, full=True):
    """
    Standardized environment instantiation function.
    If 'n' is explicitly provided, it overrides the default batch size 
    (useful for switching between training and high-volume baseline evaluations).
    """
    # Fallback to evaluation batch size if not explicitly defined
    num_boards = n if n is not None else config["N_BOARDS_EVAL"]
    size = config["BOARD_SIZE"]

    if full:
        e = environments_fully_observable.OriginalSnakeEnvironment(num_boards, size)
    else:
        mask = config["MASK SIZE"]
        e=environments_partially_observable.OriginalSnakeEnvironment(num_boards, size, mask)
        
    return e

# ------------------------------------------------------------------------------------------
#                                         DQN
# ------------------------------------------------------------------------------------------

class VectorizedReplayBuffer:
    """Efficient cyclic buffer for multi-board batch state storage."""
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        
        self.idx = 0
        self.size = 0

    def push_batch(self, states, actions, rewards, next_states):
        """Pushes a full synchronous parallel batch of entries into the buffer."""
        n = states.shape[0]     # number of parallel boards
        # Handle wrap-around index manipulation safely
        for i in range(n):
            self.states[self.idx] = states[i]
            self.actions[self.idx] = actions[i]
            self.rewards[self.idx] = rewards[i]
            self.next_states[self.idx] = next_states[i]
            
            self.idx = (self.idx + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Samples a random mini-batch of past experiences."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            tf.convert_to_tensor(self.states[indices]),
            tf.convert_to_tensor(self.actions[indices]),
            tf.convert_to_tensor(self.rewards[indices]),
            tf.convert_to_tensor(self.next_states[indices])
        )
    
def build_q_network(input_shape, action_space=4):
    """Optimal CNN architecture for small spatial grid worlds."""
    model = K.Sequential([
        K.layers.Input(shape=input_shape), # Shape: (8, 8, 4)
        
        # 1. Spatial Feature Extraction (No Pooling!)
        K.layers.Conv2D(
            filters=32, 
            kernel_size=(3, 3), 
            padding='same', 
            activation='relu',
            kernel_initializer=K.initializers.HeUniform()
        ),
        K.layers.Conv2D(
            filters=64, 
            kernel_size=(3, 3), 
            padding='same', 
            activation='relu',
            kernel_initializer=K.initializers.HeUniform()
        ),
        
        # 2. Flatten spatial maps to transition into strategic reasoning
        K.layers.Flatten(),
        
        # 3. Decision Dense Layer
        K.layers.Dense(
            64, 
            activation='relu', 
            kernel_initializer=K.initializers.HeUniform()
        ),
        
        # 4. Output Layer
        K.layers.Dense(
            action_space, 
            activation='linear',
            kernel_initializer=K.initializers.TruncatedNormal(mean=0.0, stddev=0.01),
            bias_initializer=K.initializers.Zeros()
        )
    ])
    return model

def dqn_policy(env, q_network):
    states = env.to_state()
    q_values = q_network(states, training=False).numpy()
    return np.argmax(q_values, axis=1)

def train_dqn(config, full=True):
    # 1. Initialize environment properties dynamically using the configuration mapping
    env_t = get_env(config, n=config["N_BOARDS_TRAIN"], full=full
    )
    gamma = config["GAMMA"]
    eval_freq = config["EVAL_FREQ"]
    tau= config["TAU"]

    # Dynamically extract network input shape
    state_sample = env_t.to_state()
    state_shape = state_sample.shape[1:]  
    
    # Instantiate Q-Networks and Memory
    q_network = build_q_network(state_shape)
    target_network = build_q_network(state_shape)
    target_network.set_weights(q_network.get_weights())
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config["LEARNING_RATE"])
    buffer = VectorizedReplayBuffer(capacity=config["BUFFER_CAPACITY"], state_shape=state_shape)
    
    epsilon = config["EPSILON_START"]
    states = env_t.to_state() 
    
    # Metrics history dictionary
    eval_history = {
        "step": [], 
        "avg_wins":[],
        "avg_fruits": [], 
        "avg_wall_hits": [], 
        "avg_self_bites": [], 
        "avg_raw_reward": [],
        "avg_discounted_return": []  
    }
    
    print(f"Beginning DQN Training with embedded evaluation intervals every {eval_freq} steps...")
    
    for step in range(config["TRAINING_STEPS"]):
        
        # --- PERIODIC EVALUATION CYCLE ---
        if step % eval_freq == 0:
            env_eval = get_env(config, 
                n=config["N_BOARDS_EVAL"], 
                full=full
            )
            dqn_policy_fn = lambda e: dqn_policy(e, q_network)
            
            # Extract independent evaluation values (including return calculations)
            metrics = run_evaluation(env_eval, dqn_policy_fn, max_steps=100, gamma=gamma)
            
            eval_history["step"].append(step)
            eval_history["avg_wins"].append(metrics["avg_wins"])
            eval_history["avg_fruits"].append(metrics["avg_fruits"])
            eval_history["avg_wall_hits"].append(metrics["avg_wall_hits"])
            eval_history["avg_self_bites"].append(metrics["avg_self_bites"])
            eval_history["avg_raw_reward"].append(metrics["avg_raw_reward"])
            eval_history["avg_discounted_return"].append(metrics["avg_discounted_return"])
            
            print(f"[EVAL @ STEP {step:5d}, espilon {epsilon:.3f}] Return (G0): {metrics['avg_discounted_return']:.4f} | "
                  f"Wins: {metrics['avg_wins']:.2f} | Fruits: {metrics['avg_fruits']:.2f} | Wall Hits: {metrics['avg_wall_hits']:.1f} | Bites: {metrics['avg_self_bites']:.1f}")

        # --- STANDARD TRAINING STEP LOOP ---
        if random.random() < epsilon:
            actions = np.random.choice(4, size=config["N_BOARDS_TRAIN"])
        else:
            q_values = q_network(states, training=False)
            actions = tf.argmax(q_values, axis=1).numpy()
            
        actions_input = actions[:, None] if actions.ndim == 1 else actions
        rewards_tensor = env_t.move(actions_input)
        rewards = rewards_tensor.numpy()[:, 0]
        
        next_states = env_t.to_state()
        buffer.push_batch(states, actions, rewards, next_states)
        states = next_states 
        
        if buffer.size > config["BATCH_SIZE"]:
            b_states, b_actions, b_rewards, b_next_states = buffer.sample(batch_size=config["BATCH_SIZE"])
            next_q_values = target_network(b_next_states, training=False)
            max_next_q = tf.reduce_max(next_q_values, axis=1)
            target_q_targets = b_rewards + (gamma * max_next_q)
            
            with tf.GradientTape() as tape:
                current_q_values = q_network(b_states, training=True)
                one_hot_actions = tf.one_hot(b_actions, depth=4)
                chosen_q_values = tf.reduce_sum(current_q_values * one_hot_actions, axis=1)
                huber = tf.keras.losses.Huber()
                loss = huber(target_q_targets, chosen_q_values)
                
            grads = tape.gradient(loss, q_network.trainable_variables)
            optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
            
        epsilon = max(config["EPSILON_END"], epsilon * config["EPSILON_DECAY"])
        
        # Continuous target updates (Polyak Averaging)
        online_weights = q_network.get_weights()
        target_weights = target_network.get_weights()
        new_weights = [tau * o + (1-tau) * t for o, t in zip(online_weights, target_weights)]
        target_network.set_weights(new_weights)
            
    print("Training finished! Saving model...")
    os.makedirs("models", exist_ok=True)
    if full:
        q_network.save("models/dqn_snake.keras")
    else:
        q_network.save("models/dqn_snake_partial.keras")
    
    return pd.DataFrame(eval_history)


# ------------------------------------------------------------------------------------------
#                                         PPO
# ------------------------------------------------------------------------------------------

def build_ppo_networks(input_shape, action_space=4):
    """
    Creates the dual models required for PPO using separate CNN backbones.
    """
    actor = K.Sequential([
        K.layers.Input(shape=input_shape),
        K.layers.Conv2D(
            filters=32, kernel_size=(3, 3), padding='same', activation='relu',
            kernel_initializer=K.initializers.HeUniform()
        ),
        K.layers.Conv2D(
            filters=64, kernel_size=(3, 3), padding='same', activation='relu',
            kernel_initializer=K.initializers.HeUniform()
        ),
        K.layers.Flatten(),
        K.layers.Dense(64, activation='relu', kernel_initializer=K.initializers.HeUniform()),
        
        # Safe initialization for uniform early exploration
        K.layers.Dense(
            action_space, 
            activation='softmax',
            kernel_initializer=K.initializers.TruncatedNormal(stddev=0.01),
            bias_initializer=K.initializers.Zeros()
        ) 
    ])
    
    critic = K.Sequential([
        K.layers.Input(shape=input_shape),
        K.layers.Conv2D(
            filters=32, kernel_size=(3, 3), padding='same', activation='relu',
            kernel_initializer=K.initializers.HeUniform()
        ),
        K.layers.Conv2D(
            filters=64, kernel_size=(3, 3), padding='same', activation='relu',
            kernel_initializer=K.initializers.HeUniform()
        ),
        K.layers.Flatten(),
        K.layers.Dense(64, activation='relu', kernel_initializer=K.initializers.HeUniform()),
        
        # Safe initialization for baseline evaluation values starting near 0
        K.layers.Dense(
            1, 
            activation='linear',
            kernel_initializer=K.initializers.TruncatedNormal(stddev=0.01),
            bias_initializer=K.initializers.Zeros()
        )    
    ])
    return actor, critic

def ppo_policy(env, ppo_actor):
    states = env.to_state()
    # PPO returns probabilistic matrices array profiles
    probabilities = ppo_actor(states, training=False).numpy()
    # Select the highest probability index as the optimal action choice
    return np.argmax(probabilities, axis=1)

def sample_categorical_actions(action_probs):
    """
    Samples one discrete action per parallel board using its probability distribution.
    Guarantees choices remain strictly inside valid bounds [0, 1, 2, 3].
    """
    cdf = np.cumsum(action_probs, axis=1)
    r = np.random.rand(action_probs.shape[0], 1)
    
    # Calculate raw sum
    actions = np.sum(cdf < r, axis=1)
    
    # Clip to maximum valid action index (3) to prevent out-of-bound rounding errors
    return np.clip(actions, 0, action_probs.shape[1] - 1)


def compute_log_probabilities(action_probs, actions):
    """
    Extracts the log probability of the specifically chosen actions using NumPy.
    """
    n_boards = action_probs.shape[0]
    # Advanced indexing to pull the exact probability of the action chosen on each board
    chosen_probs = action_probs[np.arange(n_boards), actions]
    # Add a tiny epsilon (1e-8) to prevent taking the log of absolute 0
    return np.log(chosen_probs + 1e-8)

def compute_gae_targets(rollout_rewards, rollout_values, last_values, gamma=0.9, lmbda=0.95):
    """
    Computes generalized advantage estimations (GAE) and discounted returns-to-go targets.
    
    Inputs are lists of length ROLLOUT_STEPS containing arrays of shape (N_BOARDS,)
    """
    rollout_steps = len(rollout_rewards)
    n_boards = rollout_rewards[0].shape[0]
    
    # Convert lists to 2D matrices of shape (ROLLOUT_STEPS, N_BOARDS)
    rewards = np.array(rollout_rewards)
    values = np.array(rollout_values)
    
    advantages = np.zeros((rollout_steps, n_boards))
    last_gae_lam = 0
    
    # Append the next-state value estimation to make calculating differences easy
    # Shape becomes (ROLLOUT_STEPS + 1, N_BOARDS)
    values_extended = np.vstack([values, last_values[None, :]])
    
    # Loop BACKWARDS from the future to the present to chain temporal dependencies
    for t in reversed(range(rollout_steps)):
        # TD Error (delta) = Reward + gamma * V(s_next) - V(s_current)
        delta = rewards[t] + gamma * values_extended[t+1] - values_extended[t]
        
        # GAE recursive chain formulation
        advantages[t] = last_gae_lam = delta + gamma * lmbda * last_gae_lam
        
    # Target Returns-To-Go = Advantages + Critic Baseline Predictions
    discounted_returns = advantages + values
    
    return discounted_returns, advantages

def flatten_rollout(rollout_list):
    """
    Converts a list of length ROLLOUT_STEPS containing arrays of shape (N_BOARDS, ...)
    into a single unified array of shape (ROLLOUT_STEPS * N_BOARDS, ...)
    """
    arr = np.array(rollout_list)
    # Merges axes 0 and 1 together
    return arr.reshape(-1, *arr.shape[2:])

def compute_tf_log_probabilities(action_probs, actions):
    """
    Extracts the log probability of the chosen actions using TensorFlow operations.
    Input shapes: action_probs (BATCH_SIZE, 4), actions (BATCH_SIZE,)
    """
    batch_size = tf.shape(action_probs)[0]
    
    # Create rows indices: [0, 1, 2, ..., BATCH_SIZE-1]
    row_indices = tf.range(batch_size, dtype=tf.int32)
    
    # Stack row indices with action choices to create coordinates: [[0, action_0], [1, action_1], ...]
    gather_indices = tf.stack([row_indices, tf.cast(actions, tf.int32)], axis=1)
    
    # Extract the exact probabilities matching those coordinates
    chosen_probs = tf.gather_nd(action_probs, gather_indices)
    
    return tf.math.log(chosen_probs + 1e-8)

def train_ppo(config, full=True):
    # 1. Initialize vectorized environment
    env_t = get_env(config,
        n=config["N_BOARDS_TRAIN"], 
        full=full
    )
    
    state_sample = env_t.to_state()
    state_shape = state_sample.shape[1:] 
    
    # 2. Instantiate networks and optimizers
    actor, critic = build_ppo_networks(state_shape, action_space=4)
    actor_optimizer = K.optimizers.Adam(learning_rate=config["LEARNING_RATE"])
    critic_optimizer = K.optimizers.Adam(learning_rate=config["LEARNING_RATE"])
    
    eval_freq = config["EVAL_FREQ"]
    total_optimization_targets = config["TRAINING_STEPS"]
    eval_history = {
        "step": [], "avg_fruits": [], "avg_wall_hits": [], 
        "avg_self_bites": [], "avg_wins": [], "avg_raw_reward": [], "avg_discounted_return": []
    }
    
    # Initialize the parallel states
    states = env_t.to_state() # Shape: (N_BOARDS_TRAIN, 8, 8, 4)
    
    # Track the total environment steps taken across all boards
    optimization_step = 0
    print(f"Beginning PPO Training for exactly {total_optimization_targets} optimization updates...")
    
    # Outer loop runs for your total training allocation timeline
    while optimization_step < total_optimization_targets:
        
# --- PERIODIC EVALUATION CYCLE ---
        if optimization_step % 480 == 0:
            env_eval = get_env(config, 
                n=config["N_BOARDS_EVAL"], full=full
            )
            ppo_policy_fn = lambda e: np.argmax(actor(e.to_state(), training=False).numpy(), axis=1)
            metrics = run_evaluation(env_eval, ppo_policy_fn, max_steps=100, gamma=config["GAMMA"])
            
            # CRITICAL FIX: Append to EVERY single list in the dictionary simultaneously
            eval_history["step"].append(optimization_step)
            eval_history["avg_fruits"].append(metrics["avg_fruits"])
            eval_history["avg_wall_hits"].append(metrics["avg_wall_hits"])
            eval_history["avg_self_bites"].append(metrics["avg_self_bites"])
            eval_history["avg_wins"].append(metrics["avg_wins"])
            eval_history["avg_raw_reward"].append(metrics["avg_raw_reward"])
            eval_history["avg_discounted_return"].append(metrics["avg_discounted_return"])
            
            print(f"[EVAL @ UPDATE {optimization_step:5d}] Return (G0): {metrics['avg_discounted_return']:.4f} | "
                  f"Fruits: {metrics['avg_fruits']:.2f} | Wall/Bite: {metrics['avg_wall_hits']:.1f}/{metrics['avg_self_bites']:.1f}")
        # --- 1. DATA COLLECTION (ON-POLICY ROLLOUTS) ---
        rollout_states, rollout_actions, rollout_rewards, rollout_log_probs, rollout_values = [], [], [], [], []
        
        for t in range(config["ROLLOUT_STEPS"]):
            action_probs = actor(states, training=False).numpy()
            state_values = critic(states, training=False).numpy()
            
            actions = sample_categorical_actions(action_probs)
            log_probs = compute_log_probabilities(action_probs, actions)
            
            rewards_tensor = env_t.move(actions[:, None].copy()) 
            rewards = rewards_tensor.numpy()[:, 0]
            
            rollout_states.append(states)
            rollout_actions.append(actions)
            rollout_rewards.append(rewards)
            rollout_log_probs.append(log_probs)
            rollout_values.append(state_values[:, 0])
            
            states = env_t.to_state()
            
        last_values = critic(states, training=False).numpy()[:, 0]
        
        # --- 2. ADVANTAGE PROCESSING ---
        discounted_returns, advantages = compute_gae_targets(
            rollout_rewards, rollout_values, last_values, gamma=config["GAMMA"]
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        b_states = flatten_rollout(rollout_states)
        b_actions = flatten_rollout(rollout_actions)
        b_log_probs = flatten_rollout(rollout_log_probs)
        b_returns = flatten_rollout(discounted_returns)
        b_advantages = flatten_rollout(advantages)
        
        # --- 3. THE OPTIMIZATION STEP CRADLE ---
        dataset_size = b_states.shape[0]
        
        for epoch in range(config["PPO_EPOCHS"]):
            indices = np.random.permutation(dataset_size)
            
            for start in range(0, dataset_size, config["BATCH_SIZE"]):
                # Stop updating immediately if we reach the overall timeline cap mid-epoch
                if optimization_step >= total_optimization_targets:
                    break
                    
                end = start + config["BATCH_SIZE"]
                mb_idx = indices[start:end]
                
                # Fetch mini-batch slices
                mb_states = b_states[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_old_log_probs = b_log_probs[mb_idx]
                mb_advantages = b_advantages[mb_idx]
                mb_returns = b_returns[mb_idx]
                
                # Execute gradient step tapes
                with tf.GradientTape() as actor_tape:
                    new_probs = actor(mb_states, training=True)
                    new_log_probs = compute_tf_log_probabilities(new_probs, mb_actions)
                    ratios = tf.exp(new_log_probs - mb_old_log_probs)
                    surr1 = ratios * mb_advantages
                    surr2 = tf.clip_by_value(ratios, 1.0 - config["CLIP_EPSILON"], 1.0 + config["CLIP_EPSILON"]) * mb_advantages
                    entropy = -tf.reduce_mean(tf.reduce_sum(new_probs * tf.math.log(new_probs + 1e-8), axis=1))
                    actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2)) - (0.02 * entropy)
                    
                with tf.GradientTape() as critic_tape:
                    new_values = critic(mb_states, training=True)[:, 0]
                    huber = tf.keras.losses.Huber(delta=1.0)
                    critic_loss = huber(mb_returns, new_values)
                    
                # Apply changes to your weights
                actor_grads = actor_tape.gradient(actor_loss, actor.trainable_variables)
                actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
                
                critic_grads = critic_tape.gradient(critic_loss, critic.trainable_variables)
                critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
                
                # Increment the true optimization step counter
                optimization_step += 1
                
    # Save network model and exit loop...
    if full:
        actor.save("models/ppo_actor_snake.keras")
    else:
        actor.save("models/ppo_actor_snake_partial.keras")
    return pd.DataFrame(eval_history)