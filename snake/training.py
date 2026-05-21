import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import random
from collections import deque

# Import your concrete environment class
# from environment import OriginalSnakeEnvironment

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
        n = states.shape[0]
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


def build_q_network(input_shape, action_space=5):
    """Creates the deep Q-network model processing the one-hot encoded state tensor."""
    model = K.Sequential([
        K.layers.Input(shape=input_shape),
        # Since the grid is fully observable and small (e.g. 7x7), a dense network 
        # or flat convolutional processing learns the localized intersections well
        K.layers.Flatten(),
        K.layers.Dense(128, activation='relu'),
        K.layers.Dense(64, activation='relu'),
        K.layers.Dense(action_space, activation='linear') # Outputs raw Q-values
    ])
    return model


def train_dqn():
    # --- HYPERPARAMETERS ---
    N_BOARDS = 128         # Moderate parallel boards configuration for stable step gathering
    BOARD_SIZE = 7         # Matching your framework assignment size
    MAX_STEPS = 50000      # Total training timesteps
    BATCH_SIZE = 64        # Neural Network update size
    GAMMA = 0.9            # Your specified discount factor
    
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.995  # Decay factor per step iteration
    TARGET_UPDATE_FREQ = 20 # Sync main weights to target network every X steps
    
    # 1. Initialize environment properties
    # Using your global helper function logic or initializing directly:
    env = OriginalSnakeEnvironment(n_boards=N_BOARDS, board_size=BOARD_SIZE)
    
    # Get shape from your to_state() one-hot categorical output channel mapping
    # boards shape: (N_BOARDS, 7, 7) -> to_state() shape: (N_BOARDS, 7, 7, 4)
    state_sample = env.to_state()
    state_shape = state_sample.shape[1:]  # (7, 7, 4)
    
    # 2. Instantiate Q-Networks and Memory
    q_network = build_q_network(state_shape)
    target_network = build_q_network(state_shape)
    target_network.set_weights(q_network.get_weights())
    
    optimizer = K.optimizers.Adam(learning_rate=0.001)
    buffer = VectorizedReplayBuffer(capacity=50000, state_shape=state_shape)
    
    epsilon = EPSILON_START
    states = env.to_state() # Get initial environmental snapshot states
    
    print("Beginning Vectorized DQN Training Pipeline...")
    
    for step in range(MAX_STEPS):
        # --- EXPLORATION VS EXPLOITATION ---
        # Evaluate actions for all parallel boards simultaneously
        if random.random() < epsilon:
            # Exploration: Pick fully random valid actions (0-4) across the batch
            actions = np.random.choice(5, size=N_BOARDS)
        else:
            # Exploitation: Forward pass to extract the highest predicted Q-values
            q_values = q_network(states, training=False)
            actions = tf.argmax(q_values, axis=1).numpy()
            
        # Execute actions vector in parallel environment
        actions_input = actions[:, None] if actions.ndim == 1 else actions
        rewards_tensor = env.move(actions_input)
        rewards = rewards_tensor.numpy()[:, 0]
        
        # Capture the transformed next states layer configurations
        next_states = env.to_state()
        
        # Push collected synchronized experiences to buffer
        buffer.push_batch(states, actions, rewards, next_states)
        states = next_states # Roll over the environmental tracker pointers
        
        # --- LEARNING / OPTIMIZATION BACKPROPAGATION ---
        if buffer.size > BATCH_SIZE:
            b_states, b_actions, b_rewards, b_next_states = buffer.sample(batch_size=BATCH_SIZE)
            
            # Predict future discounted returns using the decoupled target network
            next_q_values = target_network(b_next_states, training=False)
            max_next_q = tf.reduce_max(next_q_values, axis=1)
            
            # The Bellman Optimality Equation mapping target parameters
            target_q_targets = b_rewards + (GAMMA * max_next_q)
            
            with tf.GradientTape() as tape:
                # Get current predictions
                current_q_values = q_network(b_states, training=True)
                
                # Gather only the Q-values matching the explicitly chosen actions
                one_hot_actions = tf.one_hot(b_actions, depth=5)
                chosen_q_values = tf.reduce_sum(current_q_values * one_hot_actions, axis=1)
                
                # Compute MSE Loss
                loss = tf.reduce_mean(tf.square(target_q_targets - chosen_q_values))
                
            # Perform optimization backpropagation gradient updates
            grads = tape.gradient(loss, q_network.trainable_variables)
            optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
            
        # Decay exploration factor over time
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        # Periodically sync target networks to stabilize value approximations
        if step % TARGET_UPDATE_FREQ == 0:
            target_network.set_weights(q_network.get_weights())
            
        # Logging Progress Dashboard Window
        if step % 500 == 0:
            print(f"Step: {step:5d} | Epsilon: {epsilon:.3f} | Buffer Size: {buffer.size:5d}")
            
    # --- SAVE OPTIMAL AGENT WEIGHTS ---
    print("Training finished! Exporting structural model weights...")
    q_network.save_weights("weights/dqn_snake_weights.h5")
    print("Model successfully exported to disk.")

