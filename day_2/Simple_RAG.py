import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- NEW IMPORTS for OpenAI ---
from langchain_openai import OpenAIEmbeddings # For embeddings
from langchain_openai import ChatOpenAI       # For the LLM

# --- 1. Set up your OpenAI API Key ---
# IMPORTANT: Ensure this is set securely on Render's environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY environment variable not set. "
        "Please get your token from platform.openai.com and set it."
    )

# --- 2. Define the LLM (for text generation) using OpenAI's Chat Model ---
llm = ChatOpenAI(
    model="gpt-3.5-turbo", # A good balance of cost and performance
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY # LangChain will automatically pick this up from env var if not passed
)

# --- 3. Define the Embedding Model using OpenAIEmbeddings ---
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small", # Cost-effective and powerful embedding model
    openai_api_key=OPENAI_API_KEY # LangChain will automatically pick this up from env var if not passed
)

# --- 4. Load Data (can go back to your original longer text now) ---
raw_text = """
Python is a high-level, interpreted programming language.
It was first released in 1991 by Guido van Rossum.
Python is known for its simplicity and readability, often using indentation to define code blocks.
It supports multiple programming paradigms, including object-oriented, imperative, and functional programming.
Python has a vast standard library and a large, active community that contributes to its extensive ecosystem of third-party libraries.
It is widely used for web development (Django, Flask), data analysis (Pandas, NumPy), artificial intelligence (TensorFlow, PyTorch), scientific computing, and automation.
The Zen of Python is a collection of 19 "guiding principles" for writing computer programs that influence the design of Python.
"""

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
    print("Creating FAISS vector store with OpenAI embeddings...")
    vector_store = FAISS.from_texts(chunks, embeddings_model)
    print("FAISS vector store created successfully.")
except Exception as e:
    print(f"Error creating FAISS vector store: {e}")
    print("Ensure your OPENAI_API_KEY is correct and billing is set up on OpenAI.")
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