import streamlit as st
import numpy as np
import random
import pandas as pd
import time # For simulation of progress

# --- Environment Definition (Same as before) ---
class GridWorld:
    def __init__(self):
        self.grid_size = 3
        self.num_states = self.grid_size * self.grid_size
        self.start_state = 0
        self.goal_state = 8
        self.hole_state = 4

        self.current_state = self.start_state
        self.actions = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT"
        }
        self.num_actions = len(self.actions)

        # Rewards: default -1 per step, Goal +10, Hole -10
        self.rewards_map = self._build_rewards_map()
        
        self.transition_matrix = self._build_transition_matrix()

    def _build_rewards_map(self):
        # Rewards associated with *landing* in a state or achieving a goal/hole transition
        rewards = np.full((self.num_states, self.num_actions), -0.1) # Small penalty for each step

        # Example: Reward for entering goal state from specific adjacent states
        # The agent gets +10 for reaching the goal, and -10 for the hole.
        # This setup assumes the reward is received upon *entering* the next state.
        # For simplicity in this env, we assign specific rewards for actions leading to goal/hole
        
        # From state 5, going RIGHT to 8 (goal)
        rewards[5, 3] = 10.0
        # From state 7, going UP to 8 (goal)
        rewards[7, 0] = 10.0
        
        # From state 1, going DOWN to 4 (hole)
        rewards[1, 1] = -10.0
        # From state 3, going RIGHT to 4 (hole)
        rewards[3, 3] = -10.0
        # From state 5, going LEFT to 4 (hole)
        rewards[5, 2] = -10.0
        # From state 7, going UP to 4 (hole) -- if this action from 7 can lead to hole, which it shouldn't here.
        # Re-evaluating based on actual transitions for clarity:
        # State 0: S . .
        # State 1: . H . (state 4 is hole)
        # State 2: . . G (state 8 is goal)

        # Let's just make sure final states have their specific rewards and are terminal.
        return rewards

    def _build_transition_matrix(self):
        matrix = {}
        for s in range(self.num_states):
            matrix[s] = {}
            row, col = divmod(s, self.grid_size)
            
            # UP (0)
            matrix[s][0] = s - self.grid_size if row > 0 else s
            # DOWN (1)
            matrix[s][1] = s + self.grid_size if row < self.grid_size - 1 else s
            # LEFT (2)
            matrix[s][2] = s - 1 if col > 0 else s
            # RIGHT (3)
            matrix[s][3] = s + 1 if col < self.grid_size - 1 else s
        return matrix

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        next_state = self.transition_matrix[self.current_state][action]
        reward = self.rewards_map[self.current_state, action] # Base step reward

        done = False
        if next_state == self.goal_state:
            reward = 10.0 # Goal reward overrides step reward
            done = True
        elif next_state == self.hole_state:
            reward = -10.0 # Hole penalty overrides step reward
            done = True
        
        self.current_state = next_state
        return next_state, reward, done

# --- Q-Learning Agent (Same as before) ---
class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.8, discount_factor=0.95, exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay_rate=0.005):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate

        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.q_table[state, :])

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_future_q = np.max(self.q_table[next_state, :])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[state, action] = new_q

    def decay_exploration_rate(self, episode):
        self.exploration_rate = self.min_exploration_rate + \
                               (1.0 - self.min_exploration_rate) * \
                               np.exp(-self.exploration_decay_rate * episode) # Use 1.0 as start for consistent decay

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("Simple Q-Learning GridWorld Dashboard")

# Initialize environment and agent (once per session)
if 'env' not in st.session_state:
    st.session_state.env = GridWorld()
if 'agent' not in st.session_state:
    st.session_state.agent = QLearningAgent(st.session_state.env.num_states, st.session_state.env.num_actions)
if 'total_rewards_history' not in st.session_state:
    st.session_state.total_rewards_history = []
if 'current_episode' not in st.session_state:
    st.session_state.current_episode = 0
if 'training_running' not in st.session_state:
    st.session_state.training_running = False

# Hyperparameters (can be made interactive with st.slider if desired)
total_episodes_to_train = 5000
episodes_per_run = 50 # How many episodes to train per Streamlit loop/button click

# Display current status
st.sidebar.header("Training Controls & Status")
st.sidebar.metric("Current Episode", st.session_state.current_episode)
st.sidebar.metric("Total Episodes Target", total_episodes_to_train)
st.sidebar.metric("Exploration Rate (Epsilon)", f"{st.session_state.agent.exploration_rate:.4f}")
if st.session_state.total_rewards_history:
    st.sidebar.metric("Last Episode Reward", f"{st.session_state.total_rewards_history[-1]:.2f}")

progress_bar_container = st.sidebar.empty()
progress_bar = progress_bar_container.progress(0, text="Training Progress")

# Start/Stop Training Button
col1, col2 = st.sidebar.columns(2)
if col1.button("Start/Continue Training", disabled=st.session_state.current_episode >= total_episodes_to_train):
    st.session_state.training_running = True
    st.write("Training started/continued...")
elif col2.button("Stop Training"):
    st.session_state.training_running = False
    st.write("Training stopped.")

if st.session_state.training_running and st.session_state.current_episode < total_episodes_to_train:
    # Run a batch of episodes
    for _ in range(episodes_per_run):
        if st.session_state.current_episode >= total_episodes_to_train:
            st.session_state.training_running = False
            break # Stop if max episodes reached mid-batch

        state = st.session_state.env.reset()
        done = False
        total_reward = 0

        while not done:
            action = st.session_state.agent.choose_action(state)
            next_state, reward, done = st.session_state.env.step(action)
            st.session_state.agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        
        st.session_state.total_rewards_history.append(total_reward)
        st.session_state.agent.decay_exploration_rate(st.session_state.current_episode)
        st.session_state.current_episode += 1

        # Update progress bar
        progress_percentage = min(100, int((st.session_state.current_episode / total_episodes_to_train) * 100))
        progress_bar.progress(progress_percentage, text=f"Training Progress ({st.session_state.current_episode}/{total_episodes_to_train} Episodes)")
        
        # Give Streamlit a moment to update UI for smoother progress bar (optional, might slow down training for very fast loops)
        # time.sleep(0.001) 

    st.rerun() # Rerun Streamlit to show latest data and continue loop if training_running is True

if st.session_state.current_episode >= total_episodes_to_train:
    st.session_state.training_running = False
    st.success("Training Complete!")

# --- Dashboard Display ---
st.header("Training Progress Visualization")

if st.session_state.total_rewards_history:
    df_rewards = pd.DataFrame({
        'Episode': range(len(st.session_state.total_rewards_history)),
        'Total Reward': st.session_state.total_rewards_history
    })
    st.line_chart(df_rewards.set_index('Episode'))
else:
    st.write("Click 'Start Training' to begin and see the reward history.")

st.header("Learned Q-Table")
st.write("Q-Table values (rounded):")
# Create a DataFrame for better display
q_table_df = pd.DataFrame(st.session_state.agent.q_table, 
                          columns=st.session_state.env.actions.values(), 
                          index=[f"State {i}" for i in range(st.session_state.env.num_states)])
st.dataframe(q_table_df.style.format("{:.2f}"))

# Optional: Visualize the optimal path based on current Q-table
st.header("Optimal Path Visualization (Current Q-Table)")
st.write("Hover over states to see path choices based on current Q-table.")

grid_map = [
    "S", ".", ".",
    ".", "H", ".",
    ".", ".", "G"
]

optimal_path_state_ids = []
current_test_state = st.session_state.env.start_state
test_steps = 0
max_test_steps = 20 # Prevent infinite loop in visualization
while current_test_state != st.session_state.env.goal_state and current_test_state != st.session_state.env.hole_state and test_steps < max_test_steps:
    optimal_path_state_ids.append(current_test_state)
    action_index = np.argmax(st.session_state.agent.q_table[current_test_state, :])
    next_state, _, _ = st.session_state.env.step(action_index) # Use env.step for transitions
    current_test_state = next_state
    test_steps += 1
optimal_path_state_ids.append(current_test_state) # Add final state

# Create grid display with tooltips
grid_html = "<div style='display: grid; grid-template-columns: repeat(3, 100px); gap: 5px;'>"
for i in range(st.session_state.env.num_states):
    cell_value = grid_map[i]
    if i == st.session_state.env.start_state:
        cell_value = "S"
    elif i == st.session_state.env.goal_state:
        cell_value = "G"
    elif i == st.session_state.env.hole_state:
        cell_value = "H"
    
    bg_color = "#e0e0e0"
    text_color = "black"
    if i in optimal_path_state_ids:
        bg_color = "#aaffaa" if i != st.session_state.env.hole_state else "#ffaaaa" # Green for path, red for hole if it was on path
        if i == st.session_state.env.goal_state: bg_color = "#66ff66"

    q_values_for_state = st.session_state.agent.q_table[i, :]
    tooltip_text = f"State {i}<br>"
    for action_idx, action_name in st.session_state.env.actions.items():
        tooltip_text += f"{action_name}: {q_values_for_state[action_idx]:.2f}<br>"
    
    # Indicate best action
    best_action_idx = np.argmax(q_values_for_state)
    best_action_name = st.session_state.env.actions[best_action_idx]
    tooltip_text += f"Best Action: {best_action_name}"

    grid_html += f"""
    <div style='
        width: 100px; height: 100px; 
        border: 1px solid #ccc; 
        display: flex; justify-content: center; align-items: center; 
        font-size: 2em; font-weight: bold; 
        background-color: {bg_color}; color: {text_color};
        position: relative;
    '>
        {cell_value}
        <div style='
            visibility: hidden;
            width: 150px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px 0;
            position: absolute;
            z-index: 1;
            bottom: 125%; /* Position the tooltip above the text */
            left: 50%;
            margin-left: -75px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.8em;
            font-weight: normal;
        '>
            {tooltip_text}
            <div style='
                position: absolute;
                top: 100%;
                left: 50%;
                margin-left: -5px;
                border-width: 5px;
                border-style: solid;
                border-color: #555 transparent transparent transparent;
            '></div>
        </div>
    </div>
    """

grid_html += "</div>"
st.components.v1.html(
    f"""
    <style>
    /* Tooltip container */
    .grid-item {{
      position: relative;
      display: inline-block;
    }}

    /* Tooltip text */
    .grid-item .tooltiptext {{
      visibility: hidden;
      width: 150px;
      background-color: #555;
      color: #fff;
      text-align: center;
      border-radius: 6px;
      padding: 5px 0;
      position: absolute;
      z-index: 1;
      bottom: 125%; /* Position the tooltip above the text */
      left: 50%;
      margin-left: -75px;
      opacity: 0;
      transition: opacity 0.3s;
      font-size: 0.8em;
      font-weight: normal;
    }}

    /* Tooltip arrow */
    .grid-item .tooltiptext::after {{
      content: "";
      position: absolute;
      top: 100%;
      left: 50%;
      margin-left: -5px;
      border-width: 5px;
      border-style: solid;
      border-color: #555 transparent transparent transparent;
    }}

    /* Show the tooltip text when you hover over the tooltip container */
    .grid-item:hover .tooltiptext {{
      visibility: visible;
      opacity: 1;
    }}
    </style>
    {grid_html}
    """,
    height=350
)
st.markdown("---")
st.write("Optimal Path (based on final Q-Table):")
st.write(f"Start (0) -> {' -> '.join(map(str, optimal_path_state_ids[1:]))}")