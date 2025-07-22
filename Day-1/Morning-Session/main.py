import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables

# Import your custom tools
from tools import get_weather, calculator

# Import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEndpoint
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_core.prompts import PromptTemplate

# 1. Initialize your LLM using HuggingFaceEndpoint
# Choose a model from the Hugging Face Hub.
# Be aware that free models on the Hub might have rate limits or be slower.
# Example: Google's Flan-T5-xxl, or other smaller, faster models.
# Find a suitable repo_id (e.g., "google/flan-t5-xxl" or "HuggingFaceH4/zephyr-7b-beta")
# Note: zephyr-7b-beta is a chat model, may require a specific prompt template or chat model setup.
# For a simple text generation LLM that works well with ReAct, start with a text-to-text model like Flan-T5.


try:
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-large", # Or another valid text-generation model
        temperature=0.1,
        max_new_tokens=512, # Use max_new_tokens
        # HUGGINGFACEHUB_API_TOKEN is picked up from env vars
    )
    print("HuggingFaceEndpoint LLM initialized successfully.")
except Exception as e:
    print(f"Error initializing HuggingFaceEndpoint LLM: {e}")
    print("Please ensure HUGGINGFACEHUB_API_TOKEN is set in Render's environment variables and the repo_id is correct and suitable for text generation.")
    llm = None




if llm:
    # 2. Define your tools
    tools = [get_weather, calculator]

    # 3. Get the prompt template for the ReAct agent
    prompt_template_str = """
    You are a helpful assistant. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """
    prompt = PromptTemplate.from_template(prompt_template_str)

    # 4. Create the agent
    agent = create_react_agent(llm, tools, prompt)

    # 5. Create the AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 6. Run the agent with questions
    print("--- Running Agent Test 1: Weather ---")
    result_weather = agent_executor.invoke({"input": "What's the weather in London?"})
    print(f"Agent's Answer (Weather): {result_weather['output']}")

    print("\n--- Running Agent Test 2: Calculation ---")
    result_calc = agent_executor.invoke({"input": "What is 100 minus 25?"})
    print(f"Agent's Answer (Calculation): {result_calc['output']}")