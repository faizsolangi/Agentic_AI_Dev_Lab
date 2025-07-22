import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
# Removed HuggingFaceEndpoint import as we are mocking its output


def run_parser_example():
    print("--- Running Output Parser Example ---")

    # 1. Initialize the Output Parser
    parser = CommaSeparatedListOutputParser()

    # 2. Simulate LLM Output (Bypassing actual LLM call due to connectivity issues)
    # This string is what a working LLM would ideally return for "List 3 common fruits."
    mock_llm_output_fruits = "apple,banana,orange"
    
    # This string is what a working LLM would ideally return for "List 5 vegetables."
    mock_llm_output_vegetables = "carrot,broccoli,spinach,potato,tomato"

    try:
        # 3. Directly Parse the Mocked Output
        print("\nSimulating LLM output for 3 fruits and parsing it...")
        fruits_list = parser.parse(mock_llm_output_fruits)
        
        print("\n--- Parsed Output for Fruits ---")
        print(f"Type of parsed output: {type(fruits_list)}")
        print(f"Parsed List: {fruits_list}")

        # Example with a different query (mocked)
        print("\nSimulating LLM output for 5 vegetables and parsing it...")
        vegetables_list = parser.parse(mock_llm_output_vegetables)
        print(f"Parsed List: {vegetables_list}")

        print("\n--- Parser Example Completed Successfully ---")

    except Exception as e:
        print(f"\nAn unexpected error occurred during parsing: {e}")
        print("This error is likely due to an issue within the parser itself if the mocked input is valid.")


if __name__ == "__main__":
    run_parser_example()