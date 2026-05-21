import numpy as np

def greedy_policy(env):
    """
    Greedy Policy: Moves towards the fruit prioritizing up/down movements, then right/left.
    """
    actions = []
    heads = np.argwhere(env.boards == env.HEAD)     # gives board, x, y of heads
    fruits = np.argwhere(env.boards == env.FRUIT)
    
    head_dict = {h[0]: (h[1], h[2]) for h in heads}
    fruit_dict = {f[0]: (f[1], f[2]) for f in fruits}
    
    for b in range(env.n_boards):
        if b not in head_dict or b not in fruit_dict:
            actions.append(env.NONE)
            continue
            
        hx, hy = head_dict[b]
        fx, fy = fruit_dict[b]
        
        if fx > hx:                         #first move up/down
            actions.append(env.UP)
        elif fx < hx:
            actions.append(env.DOWN)
        elif fy > hy:                       #then move right/left    
            actions.append(env.RIGHT)
        elif fy < hy:
            actions.append(env.LEFT)
        else:
            actions.append(env.NONE)
            
    return np.array(actions, dtype=int)


def zigzag_policy(env):
    """
    Deterministic spiral path for NxN board. Covers whole board for even N.
    """
    actions = []
    heads = np.argwhere(env.boards == env.HEAD)
    head_dict = {h[0]: (h[1], h[2]) for h in heads}
    
    N = env.board_size
    # Playable boundaries (skipping the outer walls at index 0 and N-1)
    min_coord = 1
    max_coord = N - 2

    for b in range(env.n_boards):
        if b not in head_dict:
            actions.append(env.NONE)
            continue
            
        r, c = head_dict[b]
        
        # 1. Leftmost playable column: Always go UP to clear the board
        if c == min_coord:
            if r < max_coord:
                actions.append(env.UP)
            else:
                actions.append(env.RIGHT) # Reached top-left, turn right
                
        # 2. Top playable row: Always go RIGHT to reach the far right wall
        elif r == max_coord:
            if c < max_coord:
                actions.append(env.RIGHT)
            else:
                actions.append(env.DOWN) # Reached top-right, start zigzag down
                
        # 3. Internal Zig-Zag columns
        else:
            # If N-2 is odd, then columns matching its parity go DOWN
            if c % 2 == max_coord % 2:
                if r > min_coord:
                    actions.append(env.DOWN)
                else:
                    actions.append(env.LEFT) # Reached bottom, step left
            else:
                # Other columns go UP
                if r < max_coord - 1:
                    actions.append(env.UP)
                else:
                    actions.append(env.LEFT) # Reached the sub-top, step left
                    
    return np.array(actions, dtype=int)