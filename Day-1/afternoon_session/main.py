import os
from langchain_tavily import TavilySearch

tool = TavilySearch(
    max_results=5,
    topic="general",
)

try:
    print("Invoking Tavily tool for Wimbledon query...")
    wimbledon_result_dict = tool.invoke({"query": "What happened at the last wimbledon"})
    
    print(f"Type of wimbledon_result: {type(wimbledon_result_dict)}")
    print(f"Content of wimbledon_result (raw): {wimbledon_result_dict}")

    # --- FIX IS HERE: Accessing content from the dictionary structure ---
    if 'results' in wimbledon_result_dict and wimbledon_result_dict['results']:
        print("\nWimbledon Search Results:")
        for i, res in enumerate(wimbledon_result_dict['results']):
            print(f"  Result {i+1} (Title: {res.get('title', 'N/A')}):")
            print(f"    Content: {res.get('content', 'No content available')[:200]}...") # Print first 200 chars
            print(f"    URL: {res.get('url', 'N/A')}")
    else:
        print("No results found for Wimbledon query.")
    # ------------------------------------------------------------------


    print("\nInvoking Tavily tool for Euro 2024 query...")
    model_generated_tool_call = {
        "args": {"query": "euro 2024 host nation"},
        "id": "1",
        "name": "tavily",
        "type": "tool_call",
    }
    
    euro_query_args = model_generated_tool_call["args"]
    euro_result_dict = tool.invoke(euro_query_args)
    
    print(f"Type of euro_result: {type(euro_result_dict)}")
    print(f"Content of euro_result (raw): {euro_result_dict}")

    # --- FIX IS HERE: Accessing content from the dictionary structure ---
    if 'results' in euro_result_dict and euro_result_dict['results']:
        print("\nEuro 2024 Search Results:")
        for i, res in enumerate(euro_result_dict['results']):
            print(f"  Result {i+1} (Title: {res.get('title', 'N/A')}):")
            print(f"    Content: {res.get('content', 'No content available')[:200]}...") # Print first 200 chars
            print(f"    URL: {res.get('url', 'N/A')}")
    else:
        print("No results found for Euro 2024 query.")
    # ------------------------------------------------------------------


except Exception as e:
    print(f"An unexpected error occurred: {e}")
    print("Please ensure TAVILY_API_KEY is correctly set in Render's environment variables and the network is stable.")