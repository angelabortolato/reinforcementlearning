import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import random
from collections import deque
import environments_fully_observable 
import os


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
    """Creates the deep Q-network model processing the one-hot encoded state tensor."""
    model = K.Sequential([
        K.layers.Input(shape=input_shape),
        # Since the grid is fully observable and small (e.g. 8x8), a dense network 
        # or flat convolutional processing learns the localized intersections well
        K.layers.Flatten(),
        K.layers.Dense(128, activation='relu'),
        K.layers.Dense(64, activation='relu'),
        K.layers.Dense(action_space, activation='linear') # Outputs raw Q-values
    ])
    return model


def train_dqn(config):
    # 1. Initialize environment properties dynamically using the configuration mapping
    env_t = environments_fully_observable.OriginalSnakeEnvironment(
        n_boards=config["N_BOARDS_TRAIN"], 
        board_size=config["BOARD_SIZE"]
    )
    
    # Dynamically extract network input shape from one-hot state output layers
    state_sample = env_t.to_state()
    state_shape = state_sample.shape[1:]  # Automatically computes (BOARD_SIZE, BOARD_SIZE, 4)
    
    # 2. Instantiate Q-Networks and Memory
    q_network = build_q_network(state_shape)
    target_network = build_q_network(state_shape)
    target_network.set_weights(q_network.get_weights())
    
    optimizer = K.optimizers.Adam(learning_rate=config["LEARNING_RATE"])
    buffer = VectorizedReplayBuffer(capacity=config["BUFFER_CAPACITY"], state_shape=state_shape)
    
    epsilon = config["EPSILON_START"]
    states = env_t.to_state() 
    
    print(f"Beginning DQN Training on {config['N_BOARDS_TRAIN']} boards of size {config['BOARD_SIZE']}x{config['BOARD_SIZE']}...")
    
    for step in range(config["MAX_STEPS"]):
        # --- EXPLORATION VS EXPLOITATION: epsilon greedy ---
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
        
        # --- LEARNING / OPTIMIZATION BACKPROPAGATION ---
        if buffer.size > config["BATCH_SIZE"]:
            b_states, b_actions, b_rewards, b_next_states = buffer.sample(batch_size=config["BATCH_SIZE"])
            
            next_q_values = target_network(b_next_states, training=False)
            max_next_q = tf.reduce_max(next_q_values, axis=1)
            
            # Use global gamma parameter configuration safely
            target_q_targets = b_rewards + (config["GAMMA"] * max_next_q)
            
            with tf.GradientTape() as tape:
                current_q_values = q_network(b_states, training=True)
                one_hot_actions = tf.one_hot(b_actions, depth=4)
                chosen_q_values = tf.reduce_sum(current_q_values * one_hot_actions, axis=1)
                loss = tf.reduce_mean(tf.square(target_q_targets - chosen_q_values))
                
            grads = tape.gradient(loss, q_network.trainable_variables)
            optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
            
        # Decay exploration factor over time
        epsilon = max(config["EPSILON_END"], epsilon * config["EPSILON_DECAY"])
        
        # Synchronize target network periodically
        if step % config["TARGET_UPDATE_FREQ"] == 0:
            target_network.set_weights(q_network.get_weights())
            
        if step % 1000 == 0:
            print(f"Step: {step:5d} | Epsilon: {epsilon:.3f} | Buffer Size: {buffer.size:5d}")
            
    print("Training finished! Saving model...")
    os.makedirs("models", exist_ok=True)
    q_network.save("models/dqn_snake.keras")


def build_ppo_networks(input_shape, action_space=4):
    """
    Creates the dual models required for PPO:
    1. The Actor (Policy Network) outputs categorical probability distributions.
    2. The Critic (Value Network) outputs a scalar baseline state value evaluation V(s).
    """
    actor = K.Sequential([
        K.layers.Input(shape=input_shape),
        K.layers.Flatten(),
        K.layers.Dense(128, activation='relu'),
        K.layers.Dense(64, activation='relu'),
        K.layers.Dense(action_space, activation='softmax') # Guarantees valid probability distribution
    ])
    
    critic = K.Sequential([
        K.layers.Input(shape=input_shape),
        K.layers.Flatten(),
        K.layers.Dense(128, activation='relu'),
        K.layers.Dense(64, activation='relu'),
        K.layers.Dense(1, activation='linear')    # Outputs continuous expected future returns
    ])
    
    return actor, critic

def train_ppo(config):
    # 1. Initialize vectorized environment
    env_t = environments_fully_observable.OriginalSnakeEnvironment(
        n_boards=config["N_BOARDS_TRAIN"], 
        board_size=config["BOARD_SIZE"]
    )
    
    state_sample = env_t.to_state()
    state_shape = state_sample.shape[1:] # (BOARD_SIZE, BOARD_SIZE, 4)
    
    # 2. Instantiate networks and optimizers
    actor, critic = build_ppo_networks(state_shape, action_space=4)
    actor_optimizer = K.optimizers.Adam(learning_rate=config["LEARNING_RATE"])
    critic_optimizer = K.optimizers.Adam(learning_rate=config["LEARNING_RATE"])
    
    # Extract hyperparameter targets from global configuration mappings
    ROLLOUT_STEPS = config.get("ROLLOUT_STEPS", 20) # Steps collected per board per rollout phase
    PPO_EPOCHS = config.get("PPO_EPOCHS", 4)        # Gradient evaluation loops over rollout batches
    CLIP_VAL = config.get("CLIP_EPSILON", 0.2)      # PPO clip parameter bounds
    GAMMA = config["GAMMA"]
    
    states = env_t.to_state()
    print(f"Beginning Vectorized PPO Training on {config['N_BOARDS_TRAIN']} parallel boards...")
    
    # Calculate training iterations based on your target step capacity limits
    iterations = config["MAX_STEPS"] // ROLLOUT_STEPS
    
    for itr in range(iterations):
        # Memory storage lists for on-policy batch trajectory accumulation
        b_states, b_actions, b_rewards, b_old_log_probs, b_values = [], [], [], [], []
        
        # --- 1. TRAJECTORY ROLLOUT PHASE (Data Gathering) ---
        for _ in range(ROLLOUT_STEPS):
            # Compute action probability distributions using current actor weights
            probs = actor(states, training=False)
            
            # Use log-probability categorical transformations for sample collection step selection
            logits = tf.math.log(probs + 1e-10)
            actions = tf.random.categorical(logits, num_samples=1)[:, 0].numpy()
            
            # Extract log-probabilities corresponding strictly to chosen actions
            one_hot = tf.one_hot(actions, depth=4)
            old_log_probs = tf.math.log(tf.reduce_sum(probs * one_hot, axis=1) + 1e-10).numpy()
            
            # Fetch critic baseline expectations
            values = critic(states, training=False).numpy()[:, 0]
            
            # Step the vectorized environment forward
            actions_input = actions[:, None] if actions.ndim == 1 else actions
            rewards_tensor = env_t.move(actions_input)
            rewards = rewards_tensor.numpy()[:, 0]
            
            # Log trajectories
            b_states.append(states)
            b_actions.append(actions)
            b_rewards.append(rewards)
            b_old_log_probs.append(old_log_probs)
            b_values.append(values)
            
            # Progress tracking pointers update
            states = env_t.to_state()
            
        # Convert lists to high-speed numpy matrix arrays for calculation handling
        b_states = np.array(b_states).reshape(-1, *state_shape)
        b_actions = np.array(b_actions).flatten()
        b_rewards = np.array(b_rewards).reshape(ROLLOUT_STEPS, -1)
        b_old_log_probs = np.array(b_old_log_probs).flatten()
        b_values = np.array(b_values).reshape(ROLLOUT_STEPS, -1)
        
        # Calculate terminal boot-strap state values evaluation
        next_values = critic(states, training=False).numpy()[:, 0]
        
        # --- 2. ADVANTAGE AND TARGET LABELS COMPUTATION ---
        # Construct temporal difference returns (TD targets) using explicit GAE loops
        returns = np.zeros_like(b_rewards)
        last_gae = np.zeros(config["N_BOARDS_TRAIN"])
        
        for t in reversed(range(ROLLOUT_STEPS)):
            next_v = next_values if t == ROLLOUT_STEPS - 1 else b_values[t + 1]
            delta = b_rewards[t] + GAMMA * next_v - b_values[t]
            # Standard GAE lambda value can be safely parameterized to 0.95
            returns[t] = delta + b_values[t] # Simplified direct empirical value return mapping
            
        b_returns = returns.flatten()
        b_advantages = b_returns - b_values.flatten()
        # Normalize advantages across the rollout batch block to stabilize gradient descents
        b_advantages = (b_advantages - np.mean(b_advantages)) / (np.std(b_advantages) + 1e-8)
        
        # --- 3. GRADIENT OPTIMIZATION BACKPROPAGATION PHASE ---
        for _ in range(PPO_EPOCHS):
            with tf.GradientTape(persistent=True) as tape:
                # Forward passes
                new_probs = actor(b_states, training=True)
                new_values = critic(b_states, training=True)[:, 0]
                
                # Fetch new log probabilities for the executed actions vector
                one_hot_act = tf.one_hot(b_actions, depth=4)
                chosen_probs = tf.reduce_sum(new_probs * one_hot_act, axis=1)
                new_log_probs = tf.math.log(chosen_probs + 1e-10)
                
                # Compute probability ratios: r_t(theta) = pi(a|s) / pi_old(a|s)
                ratios = tf.exp(new_log_probs - b_old_log_probs)
                
                # PPO Clipped Objective Function Calculation Math
                surr1 = ratios * b_advantages
                surr2 = tf.clip_by_value(ratios, 1.0 - CLIP_VAL, 1.0 + CLIP_VAL) * b_advantages
                actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                
                # Value Objective Function (Critic Mean Squared Error Loss)
                critic_loss = tf.reduce_mean(tf.square(b_returns - new_values))
                
            # Apply gradients independently to maintain clean modular update lines
            actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
            critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
            
            actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
            critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
            del tape
            
        global_step = (itr + 1) * ROLLOUT_STEPS
        if global_step % 1000 == 0 or itr == iterations - 1:
            print(f"Global Step: {global_step:5d} | Mean Actor Loss: {float(actor_loss):.4f} | Mean Critic Loss: {float(critic_loss):.4f}")
            
    # --- 4. EXPORT PPO MODEL WEIGHTS ---
    print("PPO Optimization Finished! Exporting Actor parameters...")
    os.makedirs("models", exist_ok=True)
    actor.save("models/ppo_snake_actor.keras")
    print("Actor Policy model successfully exported to disk.")