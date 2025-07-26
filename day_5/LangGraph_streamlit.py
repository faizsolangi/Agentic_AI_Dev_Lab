import streamlit as st
import os
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from operator import add # This helps with appending lists in state updates
import uuid # For unique session IDs if using checkpointers, though not strictly needed for this simple example

# --- Configuration ---
# Ensure your OpenAI API key is set as an environment variable (recommended)
# or you can prompt the user for it in the sidebar.
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" # Uncomment and replace if not using env var

# --- LangGraph Agent Definition (Copied from previous example) ---

# Define the State of our Graph
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add] # Use 'add' to append messages
    question_asked: str
    critique_count: int
    execution_log: Annotated[list[str], add] # To log node executions for display

# --- Initialize LLM (cached for Streamlit efficiency) ---
@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

llm = get_llm()

# --- Define the Nodes ---

def call_llm_node(state: AgentState):
    current_log = state.get("execution_log", [])
    current_log.append("Executing Node: call_llm")
    
    messages = state["messages"]
    
    # If this is a retry, append a system message to guide the LLM
    if state["critique_count"] > 0:
        messages.append(
            SystemMessage(content=f"Your previous answer was too short. Please provide a more detailed answer to the question: '{state['question_asked']}'.")
        )

    response = llm.invoke(messages)
    current_log.append(f"LLM Response (raw): {response.content[:50]}...") # Log first 50 chars
    return {"messages": [response], "execution_log": current_log}

def critique_node(state: AgentState):
    current_log = state.get("execution_log", [])
    current_log.append("Executing Node: critique")
    
    ai_message = state["messages"][-1] # Get the last AI message
    critique_count = state["critique_count"]

    critique = ""
    if len(ai_message.content) < 50: # Increased length for better demo
        critique = "too_short"
        current_log.append(f"Critique: Answer is too short ({len(ai_message.content)} chars). Needs refinement.")
    else:
        critique = "satisfactory"
        current_log.append(f"Critique: Answer is satisfactory ({len(ai_message.content)} chars).")

    return {"critique_result": critique, "critique_count": critique_count + 1, "execution_log": current_log}

# --- Define the Conditional Edge Router ---

def decide_next_step(state: AgentState):
    current_log = state.get("execution_log", [])
    current_log.append("Executing Router: decide_next_step")
    
    critique_result = state.get("critique_result")
    critique_count = state.get("critique_count", 0)

    if critique_result == "satisfactory":
        current_log.append("Decision: Satisfactory, ending process.")
        return "end_process"
    elif critique_result == "too_short" and critique_count < 2: # Allow up to 2 retries
        current_log.append(f"Decision: Too short, retrying (Attempt {critique_count+1}).")
        return "call_llm"
    else:
        current_log.append("Decision: Max retries reached or other issue. Ending process.")
        return "end_process"

# --- Build the Graph (cached for Streamlit efficiency) ---
@st.cache_resource
def get_langgraph_app():
    workflow = StateGraph(AgentState)
    workflow.add_node("call_llm", call_llm_node)
    workflow.add_node("critique", critique_node)
    workflow.set_entry_point("call_llm")
    workflow.add_edge("call_llm", "critique")
    workflow.add_conditional_edges(
        "critique",
        decide_next_step,
        {
            "call_llm": "call_llm",
            "end_process": END
        }
    )
    return workflow.compile()

langgraph_app = get_langgraph_app()

# --- Streamlit UI ---
st.set_page_config(page_title="LangGraph Self-Correcting Agent", layout="wide")
st.title("LangGraph Self-Correcting LLM Agent")

st.markdown("""
This demonstration shows a LangGraph agent that attempts to answer a question.
If its answer is too short (less than 50 characters), it will critique itself and
loop back to the LLM to ask for a more detailed answer (up to 2 retries).
""")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "langgraph_state" not in st.session_state:
    st.session_state.langgraph_state = None
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "execution_log" not in st.session_state:
    st.session_state.execution_log = []

# --- OpenAI API Key Input ---
st.sidebar.header("OpenAI API Key")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    else:
        st.info("Please enter your OpenAI API key in the sidebar to use the LLM.")
        st.stop()


# --- Chat Interface ---
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask the agent a question..."):
    # Add user message to chat history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Initialize LangGraph state for this new question
    st.session_state.langgraph_state = {
        "messages": [HumanMessage(content=prompt)],
        "question_asked": prompt,
        "critique_count": 0,
        "execution_log": [] # Reset log for new question
    }
    st.session_state.current_question = prompt

    # Run LangGraph Agent
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Invoke the LangGraph app
        # LangGraph's .invoke() runs until END or an error.
        # We can't stream node-by-node directly with .invoke() here,
        # but we'll show the final result and the log.
        
        with st.spinner("Agent thinking..."):
            # The invoke call will run the graph to completion
            final_state = langgraph_app.invoke(st.session_state.langgraph_state)
            
            # Update session state with the final state
            st.session_state.langgraph_state = final_state
            st.session_state.execution_log = final_state["execution_log"]

            # Extract the final AI message
            ai_response = ""
            for msg in final_state["messages"]:
                if isinstance(msg, AIMessage):
                    ai_response = msg.content
                    break # Take the first AI message or the last one, depending on desired behavior

            full_response = ai_response
            message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})


# --- Debugging/Execution Log Section ---
st.sidebar.subheader("LangGraph Execution Log")
if st.session_state.execution_log:
    for entry in st.session_state.execution_log:
        st.sidebar.text(entry)
else:
    st.sidebar.text("No execution log yet. Ask a question to start.")

st.sidebar.subheader("Current Agent State")
if st.session_state.langgraph_state:
    st.sidebar.json(st.session_state.langgraph_state)
else:
    st.sidebar.text("Agent state will appear here after execution.")