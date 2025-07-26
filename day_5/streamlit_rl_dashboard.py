import streamlit as st
import numpy as np
import random
import pandas as pd
import time # Included for potential future use or debugging, currently not actively used in core loop

# --- Environment Definition ---
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

        self.rewards_map = self._build_rewards_map()
        self.transition_matrix = self._build_transition_matrix()

    def _build_rewards_map(self):
        # Initial step reward is 0.0, making it easier for positive rewards to propagate.
        rewards = np.full((self.num_states, self.num_actions), 0.0)
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
        reward = self.rewards_map[self.current_state, action] # Base step reward (0.0)

        done = False
        if next_state == self.goal_state:
            reward = 10.0 # Goal reward overrides step reward
            done = True
        elif next_state == self.hole_state:
            reward = -10.0 # Hole penalty overrides step reward
            done = True

        self.current_state = next_state
        return next_state, reward, done

# --- Q-Learning Agent ---
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
            # If all Q-values are the same (e.g., all zero), argmax returns the first one.
            # To break ties randomly, you could use:
            # best_actions = np.where(self.q_table[state, :] == np.max(self.q_table[state, :]))[0]
            # return random.choice(best_actions)
            return np.argmax(self.q_table[state, :])


    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_future_q = np.max(self.q_table[next_state, :])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[state, action] = new_q

    def decay_exploration_rate(self, episode):
        # Decays exploration_rate from initial (1.0) down to min_exploration_rate
        self.exploration_rate = self.min_exploration_rate + \
                               (1.0 - self.min_exploration_rate) * \
                               np.exp(-self.exploration_decay_rate * episode)

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("Simple Q-Learning GridWorld Dashboard")

# Define initial/default hyperparameters for the Q-Learning Agent
# These are the values the agent will start with if no changes are made
default_agent_params = {
    "learning_rate": 0.8,
    "discount_factor": 0.95,
    "exploration_rate": 1.0,
    "min_exploration_rate": 0.01,
    "exploration_decay_rate": 0.005
}

# --- Session State Initialization ---
# Initialize environment (only once per session)
if 'env' not in st.session_state:
    st.session_state.env = GridWorld()

# Initialize training state variables
if 'total_rewards_history' not in st.session_state:
    st.session_state.total_rewards_history = []
if 'current_episode' not in st.session_state:
    st.session_state.current_episode = 0
if 'training_running' not in st.session_state:
    st.session_state.training_running = False

# Initialize agent parameters and agent instance
# This block ensures the agent is only re-initialized if parameters change or on first run
if 'agent_params' not in st.session_state:
    st.session_state.agent_params = default_agent_params.copy()
    st.session_state.agent = QLearningAgent(st.session_state.env.num_states, st.session_state.env.num_actions,
                                            **st.session_state.agent_params)
# If agent exists but its parameters don't match stored, something is wrong, force re-init (safety)
elif not hasattr(st.session_state.agent, 'learning_rate') or \
     st.session_state.agent.learning_rate != st.session_state.agent_params["learning_rate"] or \
     st.session_state.agent.discount_factor != st.session_state.agent_params["discount_factor"] or \
     st.session_state.agent.exploration_rate != st.session_state.agent_params["exploration_rate"] or \
     st.session_state.agent.min_exploration_rate != st.session_state.agent_params["min_exploration_rate"] or \
     st.session_state.agent.exploration_decay_rate != st.session_state.agent_params["exploration_decay_rate"]:
    st.session_state.agent = QLearningAgent(st.session_state.env.num_states, st.session_state.env.num_actions,
                                            **st.session_state.agent_params)


# --- Sidebar for Hyperparameter Controls ---
st.sidebar.header("Q-Learning Hyperparameters")

# Use input widgets to get new parameter values from the user
# `key` is used to ensure widgets have unique identities across reruns, especially when values change.
new_lr = st.sidebar.slider("Learning Rate (alpha)", 0.01, 1.0, value=st.session_state.agent_params["learning_rate"], step=0.01, key='lr_slider')
new_df = st.sidebar.slider("Discount Factor (gamma)", 0.01, 0.99, value=st.session_state.agent_params["discount_factor"], step=0.01, key='df_slider')
new_exp_rate = st.sidebar.slider("Initial Exploration Rate (epsilon)", 0.0, 1.0, value=st.session_state.agent_params["exploration_rate"], step=0.01, key='exp_rate_slider')
new_min_exp_rate = st.sidebar.slider("Min Exploration Rate", 0.0, 0.1, value=st.session_state.agent_params["min_exploration_rate"], step=0.001, format="%.3f", key='min_exp_rate_slider')
new_exp_decay = st.sidebar.slider("Exploration Decay Rate", 0.0001, 0.1, value=st.session_state.agent_params["exploration_decay_rate"], format="%.4f", key='exp_decay_slider')

# Check if any parameter has been changed by the user in the sliders
params_changed = (
    new_lr != st.session_state.agent_params["learning_rate"] or
    new_df != st.session_state.agent_params["discount_factor"] or
    new_exp_rate != st.session_state.agent_params["exploration_rate"] or
    new_min_exp_rate != st.session_state.agent_params["min_exploration_rate"] or
    new_exp_decay != st.session_state.agent_params["exploration_decay_rate"]
)

# Button to apply new parameters and reset training
if params_changed:
    st.sidebar.warning("Hyperparameters changed! Click 'Reset & Apply Params' to restart training with new values.")

if st.sidebar.button("Reset & Apply Params", disabled=not params_changed):
    # Update the stored active parameters
    st.session_state.agent_params = {
        "learning_rate": new_lr,
        "discount_factor": new_df,
        "exploration_rate": new_exp_rate,
        "min_exploration_rate": new_min_exp_rate,
        "exploration_decay_rate": new_exp_decay
    }
    # Re-initialize the agent with the new parameters
    st.session_state.agent = QLearningAgent(st.session_state.env.num_states, st.session_state.env.num_actions,
                                            **st.session_state.agent_params)
    # Reset all training progress to start fresh
    st.session_state.total_rewards_history = []
    st.session_state.current_episode = 0
    st.session_state.training_running = False # Ensure training stops when params reset
    st.rerun() # Rerun to apply changes and clear dashboard state


st.sidebar.markdown("---")
st.sidebar.header("Training Controls & Status")
st.sidebar.metric("Current Episode", st.session_state.current_episode)
# You could make total_episodes_to_train and episodes_per_run interactive as well!
total_episodes_to_train = 10000
episodes_per_run = 100
max_steps_per_episode = 100 # Safety limit for each episode

st.sidebar.metric("Total Episodes Target", total_episodes_to_train)
st.sidebar.metric("Episodes Per Run", episodes_per_run)
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

# --- Main Training Loop ---
# This loop runs a batch of episodes if training is active
if st.session_state.training_running and st.session_state.current_episode < total_episodes_to_train:
    for _ in range(episodes_per_run):
        # Stop training if max episodes reached within the current batch
        if st.session_state.current_episode >= total_episodes_to_train:
            st.session_state.training_running = False
            break

        state = st.session_state.env.reset() # Reset environment for new episode
        done = False
        total_reward = 0
        steps_in_episode = 0

        # Run steps within an episode until done or max steps reached
        while not done and steps_in_episode < max_steps_per_episode:
            action = st.session_state.agent.choose_action(state)
            next_state, reward, done = st.session_state.env.step(action)
            st.session_state.agent.learn(state, action, reward, next_state) # Q-learning update
            state = next_state
            total_reward += reward
            steps_in_episode += 1

        st.session_state.total_rewards_history.append(total_reward)
        st.session_state.agent.decay_exploration_rate(st.session_state.current_episode) # Decay epsilon
        st.session_state.current_episode += 1 # Increment episode count

        # Update progress bar (can be put outside the inner loop for less frequent updates)
        progress_percentage = min(100, int((st.session_state.current_episode / total_episodes_to_train) * 100))
        progress_bar.progress(progress_percentage, text=f"Training Progress ({st.session_state.current_episode}/{total_episodes_to_train} Episodes)")

    st.rerun() # Trigger a full Streamlit rerun to update UI and continue training if needed

# Message when training is complete
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
    # Plotting smoothed average reward can be more informative for long runs
    df_rewards['Smoothed Reward (100-ep MA)'] = df_rewards['Total Reward'].rolling(window=100, min_periods=1).mean()
    st.line_chart(df_rewards[['Total Reward', 'Smoothed Reward (100-ep MA)']].set_index(df_rewards.index))
else:
    st.write("Click 'Start Training' to begin and see the reward history.")

st.header("Learned Q-Table")
st.write("Q-Table values (rounded):")
# Create a DataFrame for better display
q_table_df = pd.DataFrame(st.session_state.agent.q_table,
                          columns=st.session_state.env.actions.values(),
                          index=[f"State {i}" for i in range(st.session_state.env.num_states)])
st.dataframe(q_table_df.style.format("{:.2f}")) # Format to 2 decimal places

# --- Optimal Path Visualization ---
st.header("Optimal Path Visualization (Current Q-Table)")
st.write("Hover over states to see path choices based on current Q-table.")

# Define grid elements for display
grid_elements = {
    st.session_state.env.start_state: "S",
    st.session_state.env.goal_state: "G",
    st.session_state.env.hole_state: "H"
}

# Determine the optimal path based on the current Q-table
optimal_path_state_ids = []
current_test_state = st.session_state.env.start_state
test_steps = 0
max_test_steps = 20 # Prevent infinite loops if agent hasn't learned or gets stuck
while current_test_state != st.session_state.env.goal_state and \
      current_test_state != st.session_state.env.hole_state and \
      test_steps < max_test_steps:
    
    optimal_path_state_ids.append(current_test_state)
    
    # Choose action based on max Q-value (greedy policy)
    # Handle ties in Q-values: randomly choose among actions with max Q-value
    q_values_for_current_state = st.session_state.agent.q_table[current_test_state, :]
    best_actions_indices = np.where(q_values_for_current_state == np.max(q_values_for_current_state))[0]
    action_index = random.choice(best_actions_indices)
    
    # Simulate step to get next state for visualization (without learning)
    next_state = st.session_state.env.transition_matrix[current_test_state][action_index]
    current_test_state = next_state
    test_steps += 1
optimal_path_state_ids.append(current_test_state) # Add the final state (goal/hole/stuck)

# Generate HTML for the grid with tooltips
grid_html = "<div style='display: grid; grid-template-columns: repeat(3, 100px); gap: 5px;'>"
for i in range(st.session_state.env.num_states):
    cell_value = grid_elements.get(i, ".") # Default to '.' if not start, goal, or hole

    bg_color = "#e0e0e0" # Default background
    text_color = "black"

    # Highlight cells on the optimal path
    if i in optimal_path_state_ids:
        if i == st.session_state.env.goal_state:
            bg_color = "#66ff66" # Green for goal
        elif i == st.session_state.env.hole_state:
            bg_color = "#ffaaaa" # Red for hole
        else:
            bg_color = "#aaffaa" # Light green for path

    # Prepare tooltip text with Q-values for current state
    q_values_for_state = st.session_state.agent.q_table[i, :]
    tooltip_text = f"State {i}<br>"
    for action_idx, action_name in st.session_state.env.actions.items():
        tooltip_text += f"{action_name}: {q_values_for_state[action_idx]:.2f}<br>"

    # Add best action to tooltip
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
        cursor: help; /* Indicate hoverable element */
    '>
        {cell_value}
        <div class='tooltiptext'>
            {tooltip_text}
        </div>
    </div>
    """

grid_html += "</div>"
st.components.v1.html(
    f"""
    <style>
    /* Basic styling for tooltip - embedded for simplicity with st.components.v1.html */
    .tooltiptext {{
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
    .tooltiptext::after {{
      content: "";
      position: absolute;
      top: 100%;
      left: 50%;
      margin-left: -5px;
      border-width: 5px;
      border-style: solid;
      border-color: #555 transparent transparent transparent;
    }}

    /* Show the tooltip text when you hover over the container div */
    div[style*="justify-content: center"]:hover .tooltiptext {{
      visibility: visible;
      opacity: 1;
    }}
    </style>
    {grid_html}
    """,
    height=350
)
st.markdown("---")
st.write("Optimal Path (based on current Q-Table):")
st.write(f"Start (0) -> {' -> '.join(map(str, optimal_path_state_ids[1:]))}")