import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_huggingface import HuggingFaceEndpoint # Using your current LLM setup

# --- Environment Variable Check (for Hugging Face) ---
# Ensure HUGGINGFACEHUB_API_TOKEN is set in Render's environment variables.
# This code assumes it's already set and accessible.
# If you are using a different LLM (e.g., OpenAI or Google), adjust accordingly:
# os.environ["OPENAI_API_KEY"] = "your_openai_key"
# os.environ["GOOGLE_API_KEY"] = "your_google_key"
# -----------------------------------------------------


def run_parser_example():
    print("--- Running Output Parser Example ---")

    # 1. Initialize the Output Parser
    # This parser will convert a comma-separated string into a list.
    parser = CommaSeparatedListOutputParser()

    # 2. Create a Prompt Template
    # We include instructions for the LLM from the parser itself to guide its output.
    prompt = PromptTemplate(
        template="List {num} common {item_type}s.\n{format_instructions}",
        input_variables=["num", "item_type"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # 3. Initialize the LLM
    # Replace with your preferred LLM if Hugging Face continues to be an issue.
    # Make sure to set the corresponding API key in Render's environment variables.
    try:
        llm = HuggingFaceEndpoint(
            repo_id="google/flan-t5-small", # A smaller, faster model for testing
            temperature=0.7,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN") # Explicitly pass if needed, though usually auto-detected
        )
        print(f"Using LLM: {llm.repo_id}")

        # 4. Create a LangChain Chain
        # This chain connects the prompt, the LLM, and the parser.
        chain = prompt | llm | parser

        # 5. Invoke the Chain and Get Parsed Output
        print("\nRequesting a list of 3 fruits...")
        fruits_list = chain.invoke({"num": "3", "item_type": "fruit"})
        
        print("\n--- Parsed Output ---")
        print(f"Type of parsed output: {type(fruits_list)}")
        print(f"Parsed List: {fruits_list}")

        # Example with a different query
        print("\nRequesting a list of 5 vegetables...")
        vegetables_list = chain.invoke({"num": "5", "item_type": "vegetable"})
        print(f"Parsed List: {vegetables_list}")

    except Exception as e:
        print(f"\nError during LLM or parsing: {e}")
        print("Please ensure your LLM API key is correctly set and the model is accessible.")
        print("For local testing without a working LLM, you could simulate an LLM response:")
        print("e.g., `raw_llm_output = 'apple,banana,orange'` then `fruits_list = parser.parse(raw_llm_output)`")


if __name__ == "__main__":
    run_parser_example()