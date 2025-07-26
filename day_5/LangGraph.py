import os
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from operator import add # This helps with appending lists in state updates

# Ensure your OpenAI API key is set
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
# Or ensure it's loaded from your Streamlit app's environment variable

# Define the State of our Graph
# The 'messages' list will accumulate our conversation.
# 'question_asked' will store the original question.
# 'critique_count' will track how many times we've tried to self-correct.
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add] # Use 'add' to append messages
    question_asked: str
    critique_count: int

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# --- Define the Nodes ---

# Node 1: Call the LLM to generate an initial answer or refine a previous one
def call_llm_node(state: AgentState):
    print("---NODE: call_llm_node---")
    messages = state["messages"]
    
    # If this is a retry, append a system message to guide the LLM
    if state["critique_count"] > 0:
        messages.append(
            SystemMessage(content=f"Your previous answer was too short. Please provide a more detailed answer to the question: '{state['question_asked']}'.")
        )

    response = llm.invoke(messages)
    return {"messages": [response]} # Return the new AI message

# Node 2: Critique the LLM's answer
def critique_node(state: AgentState):
    print("---NODE: critique_node---")
    ai_message = state["messages"][-1] # Get the last AI message
    question = state["question_asked"]
    critique_count = state["critique_count"]

    critique = ""
    # Simple critique: check length
    if len(ai_message.content) < 20:
        critique = "too_short"
        print(f"Critique: Answer is too short ({len(ai_message.content)} chars). Needs refinement.")
    else:
        critique = "satisfactory"
        print(f"Critique: Answer is satisfactory ({len(ai_message.content)} chars).")

    # Increment critique count
    return {"critique_result": critique, "critique_count": critique_count + 1}


# --- Define the Conditional Edge Router ---

# This function determines the next step based on the critique result
def decide_next_step(state: AgentState):
    print("---ROUTER: decide_next_step---")
    critique_result = state.get("critique_result")
    critique_count = state.get("critique_count", 0)

    if critique_result == "satisfactory":
        return "end_process" # Go to END
    elif critique_result == "too_short" and critique_count < 3: # Allow up to 2 retries
        return "call_llm" # Loop back to call the LLM again
    else:
        # If too many retries or other issues, just end
        print("Max retries reached or other issue. Ending process.")
        return "end_process"

# --- Build the Graph ---

# 1. Initialize StateGraph with the defined state schema
workflow = StateGraph(AgentState)

# 2. Add Nodes
workflow.add_node("call_llm", call_llm_node)
workflow.add_node("critique", critique_node)

# 3. Set the Entry Point
# The process starts by calling the LLM
workflow.set_entry_point("call_llm")

# 4. Define Edges
# After calling the LLM, always go to the critique node
workflow.add_edge("call_llm", "critique")

# After the critique node, use the conditional edge to decide
workflow.add_conditional_edges(
    "critique",       # From the 'critique' node
    decide_next_step, # Use this function to decide next step
    {                 # Map the function's outputs to node names
        "call_llm": "call_llm",       # If "call_llm" is returned, go back to 'call_llm' node
        "end_process": END            # If "end_process" is returned, stop
    }
)

# 5. Compile the Graph
app = workflow.compile()

# --- Run the Graph ---

print("\n--- Running Attempt 1: Short Answer Expected ---")
initial_state_1 = {
    "messages": [HumanMessage(content="What is a very short answer to: What is the capital of France?")],
    "question_asked": "What is the capital of France?",
    "critique_count": 0 # Start count at 0
}
result_1 = app.invoke(initial_state_1)
print("\nFinal Result 1:")
for message in result_1["messages"]:
    if isinstance(message, AIMessage):
        print(f"AI: {message.content}")

print("\n--- Running Attempt 2: Longer Answer Expected ---")
initial_state_2 = {
    "messages": [HumanMessage(content="Describe the capital of France in 50 words.")],
    "question_asked": "Describe the capital of France in 50 words.",
    "critique_count": 0
}
result_2 = app.invoke(initial_state_2)
print("\nFinal Result 2:")
for message in result_2["messages"]:
    if isinstance(message, AIMessage):
        print(f"AI: {message.content}")

print("\n--- Visualize the graph (requires pygraphviz or pydot) ---")
# Optional: To visualize, you might need to install graphviz and pygraphviz
# pip install pygraphviz graphviz
# If you get errors, try pydot instead: pip install pydot graphviz
# And ensure Graphviz is installed on your system (e.g., via apt-get, brew, or exe installer)
try:
    from IPython.display import Image, display
    graph_img = app.get_graph().draw_png()
    display(Image(graph_img))
    with open("langgraph_example.png", "wb") as f:
        f.write(graph_img)
    print("Graph saved as langgraph_example.png")
except Exception as e:
    print(f"Could not visualize graph: {e}")
    print("You might need to install graphviz and pygraphviz (or pydot) and ensure Graphviz is on your system PATH.")