import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Import Hugging Face components for API calls
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub # For LLM generation via Inference API

# --- 1. Set up your Hugging Face API Token ---
# IMPORTANT: Store this securely in Render's environment variables (e.g., HUGGINGFACEHUB_API_TOKEN)
# DO NOT hardcode your API key directly in the script for production!
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError(
        "HUGGINGFACEHUB_API_TOKEN environment variable not set. "
        "Please get your token from huggingface.co/settings/tokens and set it."
    )

# --- 2. Define the LLM (for text generation) ---
# Using a general text generation model from Hugging Face Hub's Inference API
# Add the 'task' parameter to resolve the ValidationError
llm = HuggingFaceHub(
    repo_id="Qwen/Qwen2.5-72B-Instruct", # A good general-purpose model for RAG QA
    model_kwargs={"temperature": 0.5, "max_length": 512},
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    task="text2text-generation" # <--- ADDED THIS LINE
)

# --- 3. Define the Embedding Model (for converting text to vectors) ---
# Using HuggingFace Inference API for embeddings.
embeddings_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=HUGGINGFACEHUB_API_TOKEN,
    model_name="BAAI/bge-small-en-v1.5" # Excellent open-source embedding model
)


# --- 4. Load Data (Keeping it short for quicker testing) ---
raw_text = """
Python is a high-level, interpreted programming language.
It was first released in 1991 by Guido van Rossum.
Python is known for its simplicity and readability, often using indentation to define code blocks."""

# --- 5. Text Splitting ---
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = text_splitter.split_text(raw_text)

# --- 6. Create Vector Store (FAISS) ---
try:
    print("Creating FAISS vector store with Hugging Face Inference API embeddings...")
    vector_store = FAISS.from_texts(chunks, embeddings_model)
    print("FAISS vector store created successfully.")
except Exception as e:
    print(f"Error creating FAISS vector store: {e}")
    print("Ensure your HUGGINGFACEHUB_API_TOKEN is correct and the models are available via Inference API.")
    exit()

# --- 7. Set up RetrievalQA Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
    verbose=True
)

# --- 8. Query the RAG System ---
query = "What is Python known for?"
print(f"\nQuery: {query}")
response = qa_chain.invoke({"query": query})
print("\n--- Response ---")
print(response["result"])
print("\n--- Source Documents ---")
for doc in response["source_documents"]:
    print(f"- Content: {doc.page_content[:100]}...")

query = "When was Python first released and by whom?"
print(f"\nQuery: {query}")
response = qa_chain.invoke({"query": query})
print("\n--- Response ---")
print(response["result"])
print("\n--- Source Documents ---")
for doc in response["source_documents"]:
    print(f"- Content: {doc.page_content[:100]}...")

print("\nScript finished.")