import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import animation
from IPython.display import HTML


def animate_policy(env, policy_fn, policy_name="Baseline", max_steps=100, full=True):
    
    # Helper function to mask a board based on head location
    def get_masked_board(board_matrix):
        # Find where the head is on this board layout
        head_coords = np.argwhere(board_matrix == env.HEAD)
        if len(head_coords) == 0:
            return board_matrix.copy() # Fallback if head isn't found
        
        hx, hy = head_coords[0]
        masked_view = np.ones_like(board_matrix) * 5 # Fill everything with 5 (Gray)
        
        # Calculate the bounding box of visibility based on mask_size
        x_min = max(0, hx - env.mask_size)
        x_max = min(env.board_size, hx + env.mask_size + 1)
        y_min = max(0, hy - env.mask_size)
        y_max = min(env.board_size, hy + env.mask_size + 1)
        
        # Copy over ONLY the visible window from the true board configuration
        masked_view[x_min:x_max, y_min:y_max] = board_matrix[x_min:x_max, y_min:y_max]
        return masked_view

    # Store initial frame
    if full:
        history_boards = [env.boards[0].copy()]
    else:
        history_boards = [get_masked_board(env.boards[0])]

    for step in range(max_steps):
        actions = policy_fn(env)
        
        if actions.ndim == 1:
            actions_input = actions[:, None]
        else:
            actions_input = actions
            
        _ = env.move(actions_input)
        
        # Save frame based on visibility flag
        if full:
            history_boards.append(env.boards[0].copy())
        else:
            history_boards.append(get_masked_board(env.boards[0]))

    # ANIMATION ROUTINE
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    
    # Updated to include gray for masked tiles
    cmap = mcolors.ListedColormap(['black', 'white', 'red', 'limegreen', 'darkgreen', 'gray'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    im = ax.imshow(history_boards[0], origin="lower", cmap=cmap, norm=norm)
    
    def update_frame(i):
        im.set_array(history_boards[i])
        visibility_title = "Fully Observable" if full else f"POMDP (Mask Radius: {env.mask_size})"
        ax.set_title(f"{policy_name} — Step {i}\n({visibility_title})", fontsize=10, fontweight="bold")
        return [im]
        
    ani = animation.FuncAnimation(fig, update_frame, frames=len(history_boards), interval=250, blit=True)
    plt.close(fig) 
    
    return HTML(ani.to_jshtml())


def run_evaluation(env, policy_fn, policy_name="Baseline", max_steps=100, gamma=0.9, verbose=False):
    """Simulates evaluation boards and calculates both structural and mathematical returns."""
    history_reward = np.zeros((env.n_boards, max_steps))
    wins = np.zeros(env.n_boards)
    fruits_eaten = np.zeros(env.n_boards)
    wall_hits = np.zeros(env.n_boards)
    self_bites = np.zeros(env.n_boards)

    for step in range(max_steps):
        actions = policy_fn(env)
        actions_input = actions[:, None] if actions.ndim == 1 else actions
        rewards_tensor = env.move(actions_input)
        step_rewards = rewards_tensor.numpy()[:, 0]
        
        history_reward[:, step] = step_rewards
        wins += (step_rewards == env.WIN_REWARD)
        fruits_eaten += (step_rewards == env.FRUIT_REWARD)
        wall_hits += (step_rewards == env.HIT_WALL_REWARD)
        self_bites += (step_rewards == env.ATE_HIMSELF_REWARD)

    # --- MATHEMATICAL RETURN CALCULATION (G_0) ---
    discount_factors = gamma ** np.arange(max_steps)
    discounted_rewards = history_reward * discount_factors  # Shape: (n_boards, max_steps)
    cumulative_discounted_return = np.cumsum(discounted_rewards, axis=1)
    final_discounted_return_per_board = cumulative_discounted_return[:, -1]

    avg_wins = np.mean(wins)
    avg_fruits = np.mean(fruits_eaten)
    avg_wall_hits = np.mean(wall_hits)
    avg_self_bites = np.mean(self_bites)
    avg_raw_reward = np.mean(np.sum(history_reward, axis=1))
    avg_discounted_return = np.mean(final_discounted_return_per_board)

    if verbose:
        # Display Batch Metrics Summary Panel
        print(f"\n==========================================")
        print(f" METRICS ({env.n_boards} BOARDS): {policy_name.upper()}")
        print(f"==========================================")
        print(f"Total Combined Wins:      {np.sum(wins)}")
        print(f"Avg Wins Per Board:       {avg_wins:.4f}")
        print(f"Avg Fruits Eaten:         {avg_fruits:.2f}")
        print(f"Avg Total Wall Hits:        {avg_wall_hits:.2f}")
        print(f"Avg Total Self Bites:     {avg_self_bites:.2f}")
        print(f"Avg Total Raw Reward:     {avg_raw_reward:.2f}")
        print(f"Avg Cumulative Return (G): {avg_discounted_return:.4f} (with gamma={gamma})")
        print(f"==========================================\n")

    return {
        "avg_wins": avg_wins,
        "avg_fruits": avg_fruits,
        "avg_wall_hits": avg_wall_hits,
        "avg_self_bites": avg_self_bites,
        "avg_raw_reward": avg_raw_reward,
        "avg_discounted_return": avg_discounted_return
    }


def plot_RL_vs_baselines(RL_experiments, baseline_metrics):
    """
    Visualizes learning curves across multiple RL training runs and baseline metrics
    using an optimized 3x2 grid layout. Distinguishes multiple RL runs using 
    stylistic line variations within panel color themes.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    # 1. Setup unique line modifications for the 2 RL policies
    # Policy 1 gets a solid line, Policy 2 gets a slight dash-gap or minor alpha shift
    rl_styles = ['-', '--'] 
    rl_alphas = [1.0, 0.7]   # Slightly fades the second run for clarity
    
    rl_style_map = {}
    for idx, exp_name in enumerate(RL_experiments.keys()):
        s_idx = idx % len(rl_styles)
        rl_style_map[exp_name] = (rl_styles[s_idx], rl_alphas[s_idx])

    # 2. Setup styles for your 2 baselines 
    available_base_styles = [':', '-.']  # Dotted and Dash-dot look clean for references
    baseline_styles = {}
    for idx, base_name in enumerate(baseline_metrics.keys()):
        style_idx = idx % len(available_base_styles)
        baseline_styles[base_name] = available_base_styles[style_idx]
    
    metrics_to_plot = [
        ("avg_discounted_return", "Discounted Return ($G_0$)", "purple"),
        ("avg_fruits", "Average Fruits Eaten", "green"),
        ("avg_wins", "Average Wins Achieved", "orange"), 
        ("avg_wall_hits", "Wall Hits Per Episode", "red"),
        ("avg_self_bites", "Self-Bites Per Episode", "brown"),
        ("avg_raw_reward", "Total Raw Step Reward", "blue")
    ]
    
    for idx, (col_name, title, base_color) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # 3. Plot each RL training experiment curve with unique line styles
        for exp_name, df in RL_experiments.items():
            if col_name in df.columns:
                line_style, alpha_val = rl_style_map[exp_name]
                ax.plot(
                    df["step"], 
                    df[col_name], 
                    label=f"{exp_name}", 
                    color=base_color,
                    linestyle=line_style,
                    alpha=alpha_val,
                    linewidth=2.5
                )
            
        # 4. Draw horizontal baseline threshold control lines in a neutral dark slate color
        for base_name, stats in baseline_metrics.items():
            if col_name in stats:
                ax.axhline(
                    y=stats[col_name], 
                    linestyle=baseline_styles[base_name], 
                    color="#333333",  # Dark charcoal gray to prevent color bleeding
                    linewidth=2.0,
                    alpha=0.85,
                    zorder=2,         # Keeps them behind the main training lines
                    label=f"Baseline ({base_name})"
                )
                
        # Formatting adjustments
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Optimization Steps", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)
        
        # Use a symlog scale on panels where curves and baselines are drastically compressed near 0
        if col_name in ["avg_wins", "avg_fruits"]:
            ax.set_yscale('symlog', linthresh=0.1)
            
        ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="none", fontsize=9)
            
    plt.tight_layout()
    plt.show()