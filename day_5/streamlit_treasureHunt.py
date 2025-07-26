import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os

# --- 1. Game Environment Definition ---
class TreasureHuntGame:
    def __init__(self):
        self.locations = {
            "clearing": {
                "description": "You are in a sunny forest clearing. Sunlight filters through the leaves. Paths lead NORTH and EAST.",
                "exits": {"north": "forest_path", "east": "riverbank"},
                "items": []
            },
            "forest_path": {
                "description": "A narrow, winding path through dense trees. You hear distant animal sounds. It leads NORTH to a dark cave and SOUTH back to the clearing.",
                "exits": {"north": "dark_cave", "south": "clearing"},
                "items": []
            },
            "dark_cave": {
                "description": "It's cold and damp in this dark cave. You can barely see a glint of something shiny on the ground. The only exit is SOUTH.",
                "exits": {"south": "forest_path"},
                "items": ["shiny key"]
            },
            "riverbank": {
                "description": "You're by a gently flowing river. The water looks cool and inviting. Paths lead WEST back to the clearing and EAST towards rugged mountains.",
                "exits": {"west": "clearing", "east": "mountain_pass"},
                "items": []
            },
            "mountain_pass": {
                "description": "You're on a steep mountain pass. The wind howls around you. Ahead, you see an old, wooden CHEST. The path leads WEST.",
                "exits": {"west": "riverbank"},
                "items": ["old wooden chest"] # This is the treasure, but needs the key
            }
        }
        self.current_location_key = "clearing"
        self.inventory = []
        self.game_over = False
        self.win = False
        self.treasure_found = False

    def reset(self):
        self.current_location_key = "clearing"
        self.inventory = []
        self.game_over = False
        self.win = False
        self.treasure_found = False
        return self.get_current_state_description()

    def get_current_state_description(self):
        location = self.locations[self.current_location_key]
        desc = location["description"]
        if self.inventory:
            desc += f"\n\nIn your inventory: {', '.join(self.inventory)}."
        return desc

    def _parse_llm_action(self, llm_response: str):
        # Simplistic parsing: look for keywords
        llm_response = llm_response.lower()

        if "go north" in llm_response: return ("go", "north")
        if "go south" in llm_response: return ("go", "south")
        if "go east" in llm_response: return ("go", "east")
        if "go west" in llm_response: return ("go", "west")

        if "pick up" in llm_response or "take" in llm_response:
            if "shiny key" in llm_response:
                return ("pick_up", "shiny key")
            # Added a more flexible check for "shiny item" if "shiny key" is available in the current location
            elif "shiny item" in llm_response and "shiny key" in self.locations[self.current_location_key]["items"]:
                return ("pick_up", "shiny key")
            elif "old wooden chest" in llm_response:
                # While a chest isn't typically "picked up", the LLM might try this.
                # We handle it here and let the game logic give the specific feedback.
                return ("pick_up", "old wooden chest")

        if "open chest" in llm_response and "chest" in llm_response:
            return ("open", "chest")
            
        return ("unknown", None)

    def take_action(self, llm_response: str):
        action_type, target = self._parse_llm_action(llm_response)
        feedback = ""
        reward = 0 # Simple reward for finding treasure

        current_location = self.locations[self.current_location_key]

        if self.game_over:
            return "The game is over. Please start a new game.", 0, True

        if action_type == "go":
            if target in current_location["exits"]:
                self.current_location_key = current_location["exits"][target]
                feedback = f"You went {target}. " + self.get_current_state_description()
            else:
                feedback = f"You cannot go {target} from here. Try a different direction."
        elif action_type == "pick_up":
            if target == "shiny key" and "shiny key" in current_location["items"]:
                self.inventory.append(target)
                current_location["items"].remove(target)
                feedback = f"You picked up the {target}."
                feedback += "\nThis might be useful."
            elif target == "old wooden chest": # Can't pick up the chest
                feedback = "You can't pick up the entire chest. You need to open it."
            else:
                feedback = f"There is no {target} here to pick up."
        elif action_type == "open" and target == "chest":
            if self.current_location_key == "mountain_pass":
                if "shiny key" in self.inventory:
                    feedback = "You used the shiny shiny key to open the old wooden chest! Inside, you find a dazzling pile of gold and jewels! **YOU WIN!**"
                    reward = 100 # Huge reward for winning
                    self.game_over = True
                    self.win = True
                    self.treasure_found = True
                else:
                    feedback = "The chest is locked. You need a key."
            else:
                feedback = "There is no chest here to open."
        else:
            feedback = "I don't understand that action. Please be specific, like 'go north', 'pick up shiny key', or 'open chest'. Remember to use these exact phrases."

        return feedback, reward, self.game_over

# --- 2. Streamlit App Setup ---
st.set_page_config(layout="wide", page_title="LLM Treasure Hunt")
st.title("LLM Agent Treasure Hunt")

# --- OpenAI API Key Input ---
st.sidebar.header("OpenAI API Key")
# Check for API key in environment variable first
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    else:
        st.info("Please enter your OpenAI API key in the sidebar to start the game.")
        st.stop() # Stop execution until key is provided

# Initialize LLM (only once per session)
if "llm" not in st.session_state or st.session_state.llm_key != openai_api_key:
    st.session_state.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=openai_api_key)
    st.session_state.llm_key = openai_api_key # Store key to detect changes

# Initialize game environment and history
if "game_env" not in st.session_state:
    st.session_state.game_env = TreasureHuntGame()
    st.session_state.game_log = [] # Stores (action, feedback) tuples
    st.session_state.llm_chat_history = [] # For LLM's internal context

# --- Game Controls ---
col1, col2 = st.columns([1, 4])

with col1:
    if st.button("Start New Game"):
        st.session_state.game_env = TreasureHuntGame()
        st.session_state.game_log = []
        st.session_state.llm_chat_history = []
        # UPDATED SYSTEM MESSAGE: More explicit about required actions
        st.session_state.llm_chat_history.append(
            SystemMessage(content="You are an adventurer exploring a text-based treasure hunt. Your goal is to find the hidden treasure. You will be given descriptions of your current location and inventory. You must respond with clear, concise actions. Valid actions are: 'go north', 'go south', 'go east', 'go west', 'pick up shiny key', 'open chest'. Respond ONLY with one of these exact actions. Do not include any other text, conversation, or explanations. If you find the treasure, celebrate!"))
        st.session_state.llm_chat_history.append(
            HumanMessage(content=st.session_state.game_env.get_current_state_description())
        )
        st.rerun() # Rerun to update the display

    if st.button("Get LLM's Next Action", disabled=st.session_state.game_env.game_over):
        if not st.session_state.llm_chat_history: # Ensure game started
            st.warning("Please start a new game first!")
        else:
            # Get LLM's response
            with st.spinner("LLM is thinking..."):
                ai_response = st.session_state.llm.invoke(st.session_state.llm_chat_history)
                llm_action_text = ai_response.content.strip()

            st.write(f"LLM's raw action: `{llm_action_text}`") # For debugging

            # Process action in game environment
            feedback, reward, game_over = st.session_state.game_env.take_action(llm_action_text)

            # Update game log and LLM chat history
            st.session_state.game_log.append({"action": llm_action_text, "feedback": feedback})
            st.session_state.llm_chat_history.append(HumanMessage(content=llm_action_text)) # LLM's action is part of history
            st.session_state.llm_chat_history.append(SystemMessage(content=feedback)) # Environment feedback is system message

            if game_over:
                if st.session_state.game_env.win:
                    st.balloons()
                    st.success("Congratulations! The LLM found the treasure!")
                else:
                    st.error("The game is over.")
                st.session_state.game_log.append({"action": "--- Game Over ---", "feedback": "Game finished."})
            st.rerun() # Rerun to update display

# --- Game Display ---
with col2:
    st.subheader("Current Game State:")
    if st.session_state.game_log:
        last_action_feedback = st.session_state.game_log[-1]["feedback"]
    else:
        last_action_feedback = st.session_state.game_env.get_current_state_description() # Initial description

    st.markdown(f"**Location:** {st.session_state.game_env.current_location_key.replace('_', ' ').title()}")
    st.markdown(f"**Description:** {last_action_feedback}")
    st.markdown(f"**Inventory:** {', '.join(st.session_state.game_env.inventory) if st.session_state.game_env.inventory else 'Empty'}")
    st.markdown("---")

    st.subheader("Game Log:")
    if not st.session_state.game_log:
        st.write("Click 'Start New Game' to begin!")
    else:
        for entry in reversed(st.session_state.game_log): # Show most recent at top
            st.json(entry, expanded=False)

    if st.session_state.game_env.game_over:
        st.markdown(f"**Game Over! Status: {'WIN' if st.session_state.game_env.win else 'LOST'}.**")