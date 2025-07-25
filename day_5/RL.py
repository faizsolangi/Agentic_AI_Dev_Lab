import numpy as np
import random

# --- Environment Definition ---
class GridWorld:
    def __init__(self):
        # 3x3 grid:
        # 0 1 2
        # 3 4 5
        # 6 7 8
        self.grid_size = 3
        self.num_states = self.grid_size * self.grid_size
        self.start_state = 0 # Top-left corner
        self.goal_state = 8  # Bottom-right corner
        self.hole_state = 4  # Center

        self.current_state = self.start_state
        self.actions = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT"
        }
        self.num_actions = len(self.actions)

        # Rewards: default -1 per step, Goal +10, Hole -10
        self.rewards = np.full((self.num_states, self.num_actions), -1.0)
        
        # Set specific rewards
        # Goal state actions (reaching goal gives +10)
        self.rewards[5, 3] = 10.0 # From state 5, going RIGHT to 8 (goal)
        self.rewards[7, 0] = 10.0 # From state 7, going UP to 8 (goal)
        
        # Hole state actions (falling into hole gives -10)
        self.rewards[1, 1] = -10.0 # From state 1, going DOWN to 4 (hole)
        self.rewards[3, 3] = -10.0 # From state 3, going RIGHT to 4 (hole)
        self.rewards[5, 2] = -10.0 # From state 5, going LEFT to 4 (hole)
        self.rewards[7, 0] = -10.0 # From state 7, going UP to 4 (hole, careful with goal collision if logic is complex)

        # Map state transitions for each action
        self.transition_matrix = self._build_transition_matrix()

    def _build_transition_matrix(self):
        matrix = {}
        for s in range(self.num_states):
            matrix[s] = {}
            row, col = divmod(s, self.grid_size)
            
            # UP
            if row > 0: matrix[s][0] = s - self.grid_size # Move up
            else: matrix[s][0] = s # Stay if at top edge

            # DOWN
            if row < self.grid_size - 1: matrix[s][1] = s + self.grid_size # Move down
            else: matrix[s][1] = s # Stay if at bottom edge

            # LEFT
            if col > 0: matrix[s][2] = s - 1 # Move left
            else: matrix[s][2] = s # Stay if at left edge

            # RIGHT
            if col < self.grid_size - 1: matrix[s][3] = s + 1 # Move right
            else: matrix[s][3] = s # Stay if at right edge
        return matrix

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        next_state = self.transition_matrix[self.current_state][action]
        reward = self.rewards[self.current_state, action] # Reward for taking action from current state

        done = False
        if next_state == self.goal_state:
            reward = 10.0 # Ensure goal reward is prominent
            done = True
        elif next_state == self.hole_state:
            reward = -10.0 # Ensure hole penalty is prominent
            done = True
        
        self.current_state = next_state
        return next_state, reward, done

# --- Q-Learning Agent ---
class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.8, discount_factor=0.95, exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay_rate=0.001):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate # alpha
        self.discount_factor = discount_factor # gamma
        self.exploration_rate = exploration_rate # epsilon
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate

        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        # Epsilon-greedy strategy
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, self.num_actions - 1) # Explore: choose random action
        else:
            return np.argmax(self.q_table[state, :]) # Exploit: choose best action from Q-table

    def learn(self, state, action, reward, next_state):
        # Q-learning update formula
        # Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(s',a')) - Q(s,a)]
        current_q = self.q_table[state, action]
        max_future_q = np.max(self.q_table[next_state, :])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[state, action] = new_q

    def decay_exploration_rate(self, episode):
        self.exploration_rate = self.min_exploration_rate + \
                               (self.exploration_rate - self.min_exploration_rate) * \
                               np.exp(-self.exploration_decay_rate * episode)

# --- Training Loop ---
def train_agent(env, agent, num_episodes):
    print("--- Starting Q-Learning Training ---")
    print(f"Environment: {env.grid_size}x{env.grid_size} Grid World")
    print(f"Start: {env.start_state}, Goal: {env.goal_state}, Hole: {env.hole_state}")
    print(f"Learning Rate (alpha): {agent.learning_rate}")
    print(f"Discount Factor (gamma): {agent.discount_factor}")
    print(f"Initial Exploration Rate (epsilon): {agent.exploration_rate}")
    print(f"Number of Episodes: {num_episodes}\n")

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        
        agent.decay_exploration_rate(episode)

        if (episode + 1) % (num_episodes // 10) == 0 or episode == 0:
            print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward:.2f} - Exploration Rate: {agent.exploration_rate:.4f}")
    
    print("\n--- Training Complete ---")

# --- Test Agent (Optional, for path visualization) ---
def test_agent(env, agent, num_episodes=5):
    print("\n--- Testing Learned Policy ---")
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        path = [state]
        total_reward = 0
        steps = 0

        while not done and steps < 50: # Max steps to prevent infinite loops in testing
            action = np.argmax(agent.q_table[state, :]) # Always exploit
            next_state, reward, done = env.step(action)
            
            path.append(next_state)
            total_reward += reward
            state = next_state
            steps += 1
        
        print(f"Test Episode {episode + 1}: Path: {path} - Total Reward: {total_reward:.2f} - Steps: {steps}")
        if state == env.goal_state:
            print("  Reached Goal!")
        elif state == env.hole_state:
            print("  Fell into Hole!")
        else:
            print("  Did not reach goal or hole (max steps reached).")

# --- Main Execution ---
if __name__ == "__main__":
    env = GridWorld()
    agent = QLearningAgent(env.num_states, env.num_actions,
                           learning_rate=0.8,
                           discount_factor=0.95,
                           exploration_rate=1.0,
                           min_exploration_rate=0.01,
                           exploration_decay_rate=0.005) # Adjusted decay for faster training

    num_training_episodes = 2000 # Enough episodes to converge for a 3x3 grid

    train_agent(env, agent, num_training_episodes)

    print("\n--- Final Learned Q-Table ---")
    print(np.round(agent.q_table, 2)) # Print rounded Q-table for readability

    test_agent(env, agent, num_episodes=3) # Test a few times to see the learned path
    print("\nScript finished. Check Render logs for output.")