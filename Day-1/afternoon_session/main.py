import os

from langchain_tavily import TavilySearch

tool = TavilySearch(
    max_results=5,
    topic="general",
)

try:
    print("Invoking Tavily tool for Wimbledon query...")
    wimbledon_result = tool.invoke({"query": "What happened at the last wimbledon"})
    
    # --- DEBUGGING LINES ---
    print(f"Type of wimbledon_result: {type(wimbledon_result)}")
    print(f"Content of wimbledon_result (raw): {wimbledon_result}")
    # -----------------------

    # Only try to access .content if it's an object with that attribute
    if hasattr(wimbledon_result, 'content'):
        print("Wimbledon Search Result (first 400 chars):", wimbledon_result.content[:400])
    else:
        # If it's a dictionary, access it differently
        print("Wimbledon Search Result (as dict, full):", wimbledon_result)


    print("\nInvoking Tavily tool for Euro 2024 query...")
    model_generated_tool_call = {
        "args": {"query": "euro 2024 host nation"},
        "id": "1",
        "name": "tavily",
        "type": "tool_call",
    }
    
    euro_query_args = model_generated_tool_call["args"]
    euro_result = tool.invoke(euro_query_args)
    
    # --- DEBUGGING LINES ---
    print(f"Type of euro_result: {type(euro_result)}")
    print(f"Content of euro_result (raw): {euro_result}")
    # -----------------------

    if hasattr(euro_result, 'content'):
        print("Euro 2024 Search Result (first 400 chars):", euro_result.content[:400])
    else:
        print("Euro 2024 Search Result (as dict, full):", euro_result)


except Exception as e:
    print(f"Error calling TavilySearch tool: {e}")
    print("Please ensure TAVILY_API_KEY is correctly set in Render's environment variables.")