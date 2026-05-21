import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import animation
from IPython.display import HTML


def evaluate_and_animate_policy(env, policy_fn, policy_name="Baseline", max_steps=100, 
    gamma=0.9):

    # Store history matrix array steps ONLY for board index 0 to save memory
    history_boards = [env.boards[0].copy()]
    
    # Shape: (n_boards, max_steps) to track raw rewards across all games independently
    history_reward = np.zeros((env.n_boards, max_steps))
    
    # Track metrics across the entire duration
    fruits_eaten_per_board = np.zeros(env.n_boards)
    wall_crashes_per_board = np.zeros(env.n_boards)
    self_bites_per_board = np.zeros(env.n_boards)
    wins_per_board = np.zeros(env.n_boards) 

    for step in range(max_steps):
        # Generate actions vector for all boards
        actions = policy_fn(env)
        
        # Format the actions shape properly for env.move
        if actions.ndim == 1:
            actions_input = actions[:, None]
        else:
            actions_input = actions
            
        # Execute environmental transformation step across all boards
        rewards_tensor = env.move(actions_input)
        
        # Convert tensor to 1D numpy array of shape (n_boards,)
        step_rewards = rewards_tensor.numpy()[:, 0]
        
        # Log raw rewards continuously
        history_reward[:, step] = step_rewards
        
        # Track occurrences based on environmental reward constants
        fruits_eaten_per_board += (step_rewards == env.FRUIT_REWARD)
        wall_crashes_per_board += (step_rewards == env.HIT_WALL_REWARD)
        self_bites_per_board += (step_rewards == env.ATE_HIMSELF_REWARD)
        wins_per_board += (step_rewards == env.WIN_REWARD) 
        
        # Save only board 0's layout configuration state for rendering
        history_boards.append(env.boards[0].copy())

    # --- GAMMA DISCOUNTING CALCULATIONS ---
    # Create an array of discount factors: [gamma^0, gamma^1, gamma^2, ..., gamma^(max_steps-1)]
    discount_factors = gamma ** np.arange(max_steps)
    
    # Multiply rewards at each step by their corresponding gamma^t factor
    discounted_rewards = history_reward * discount_factors  # Shape: (n_boards, max_steps)
    
    # Compute cumulative discounted return over time by taking the cumulative sum along the steps axis
    # Shape: (n_boards, max_steps) where entry (b, t) is the discounted return accumulated up to step t
    cumulative_discounted_return = np.cumsum(discounted_rewards, axis=1)
    
    # Final total discounted return per board at the final step
    final_discounted_return_per_board = cumulative_discounted_return[:, -1]
    # --------------------------------------

    # Calculate aggregate batch statistics
    total_wins_recorded = int(np.sum(wins_per_board))
    avg_wins_per_board = np.mean(wins_per_board)
    avg_fruits = np.mean(fruits_eaten_per_board)
    avg_crashes = np.mean(wall_crashes_per_board + self_bites_per_board)
    avg_total_raw_reward = np.mean(np.sum(history_reward, axis=1))
    avg_final_discounted_return = np.mean(final_discounted_return_per_board)

    # Display Batch Metrics Summary Panel
    print(f"\n==========================================")
    print(f" PARALLEL METRICS ({env.n_boards} BOARDS): {policy_name.upper()}")
    print(f"==========================================")
    print(f"Total Combined Wins:      {total_wins_recorded}")
    print(f"Avg Wins Per Board:       {avg_wins_per_board:.4f}")
    print(f"Avg Fruits Eaten:         {avg_fruits:.2f}")
    print(f"Avg Total Crashes:        {avg_crashes:.2f}")
    print(f"Avg Total Raw Reward:     {avg_total_raw_reward:.2f}")
    print(f"Avg Cumulative Return (G): {avg_final_discounted_return:.4f} (with gamma={gamma})")
    print(f"==========================================\n")

    # 2a. PLOT: Average Reward Over Time
    mean_rewards_per_step = np.mean(history_reward, axis=0)
    
    plt.figure(figsize=(7, 3))
    plt.plot(mean_rewards_per_step, label='Mean Reward', color='dodgerblue', linewidth=2)
    plt.title(f"{policy_name} — Average Reward Per Step")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

    # 2b. PLOT: Average Cumulative Discounted Return Over Time
    mean_cumulative_return_per_step = np.mean(cumulative_discounted_return, axis=0)
    
    plt.figure(figsize=(7, 3))
    plt.plot(mean_cumulative_return_per_step, label=f'Mean Cumulative Return ($\gamma$={gamma})', color='darkorchid', linewidth=2)
    plt.title(f"{policy_name} — Average Cumulative Return Over Time")
    plt.xlabel("Timestep (t)")
    plt.ylabel("Discounted Return ($G_0$)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

    # 3. ANIMATION: Render Frame Animation Setup for Board Index 0
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    
    cmap = mcolors.ListedColormap(['black', 'white', 'red', 'limegreen', 'darkgreen'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    im = ax.imshow(history_boards[0], origin="lower", cmap=cmap, norm=norm)
    
    def update_frame(i):
        im.set_array(history_boards[i])
        ax.set_title(f"{policy_name} (Board 0) — Step {i}")
        return [im]
        
    ani = animation.FuncAnimation(fig, update_frame, frames=len(history_boards), interval=250, blit=True)
    plt.close(fig) 
    
    return HTML(ani.to_jshtml())