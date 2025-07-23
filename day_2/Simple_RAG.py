from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# We will use a mock LLM for demonstration, as live LLM calls require API keys
# or a stable local setup, which can be flaky.
# In a real scenario, replace this with ChatOpenAI, ChatGoogleGenerativeAI, etc.
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

# --- 1. Define your knowledge base ---
# Let's use a short, simple text about Python programming.
raw_text = """
Python is a high-level, interpreted programming language. It was first released in 1991 by Guido van Rossum.
Python is known for its readability and simplicity, making it a popular choice for beginners.
It is widely used for web development (Django, Flask), data analysis (Pandas, NumPy), artificial intelligence,
machine learning, automation, and scientific computing.
Python's philosophy emphasizes code readability with its notable use of significant whitespace.
The Python Package Index (PyPI) hosts thousands of third-party modules for Python.
One of its key features is its large and comprehensive standard library.
"""

# --- 2. Load the document (in this simple case, we just treat the string as a document) ---
# For more complex scenarios, you'd use TextLoader('your_file.txt') or PyPDFLoader, etc.
# For this example, we'll create a single document manually.
from langchain_core.documents import Document
documents = [Document(page_content=raw_text, metadata={"source": "python_basics"})]
print(f"Original document length: {len(raw_text)} characters")

# --- 3. Split the text into smaller chunks ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,      # Max characters per chunk
    chunk_overlap=20,    # Overlap between chunks to maintain context
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(documents)
print(f"Number of chunks created: {len(chunks)}")
# print("\nFirst few chunks:")
# for i, chunk in enumerate(chunks[:3]):
#     print(f"Chunk {i+1}: {chunk.page_content[:50]}...") # Print first 50 chars of chunk

# --- 4. Create embeddings for these chunks ---
# Using a local Sentence Transformer model: all-MiniLM-L6-v2
# This model is small and efficient, perfect for local use and demonstrations.
# It will download the model the first time it runs.
print("\nInitializing Embedding Model (may download model)...")
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embedding Model initialized.")

# --- 5. Set up an in-memory Vector Store (FAISS) ---
# FAISS is excellent for local, in-memory vector storage and similarity search.
# We create it from our documents and their embeddings.
print("Creating Vector Store (FAISS) from chunks...")
vector_store = FAISS.from_documents(chunks, embeddings_model)
print("Vector Store created.")

# --- 6. Create a Retriever ---
# The vector store can be directly converted into a retriever.
# We'll retrieve the top 2 most relevant chunks.
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
print("Retriever created.")

# --- 7. Define the Prompt Template for the LLM ---
# This template tells the LLM how to use the retrieved context.
prompt_template = ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context
to answer the question. If you don't know the answer, just say that you don't know.
Do not make up any information.

Context:
{context}

Question: {question}
""")

# --- 8. Define a Mock LLM for demonstration ---
# In a real application, you'd replace this with a live LLM instance.
class MockLLM:
    def invoke(self, prompt: BaseMessage) -> AIMessage:
        # For simplicity, we'll just print the constructed prompt
        # and give a placeholder answer.
        # In a real scenario, the LLM processes this prompt and generates a real answer.
        print("\n--- Mock LLM received this prompt for generation ---")
        print(prompt.content)
        print("--------------------------------------------------")
        # Simulate an LLM response based on the prompt content for demonstration
        if "Python" in prompt.content and "Guido van Rossum" in prompt.content:
            return AIMessage(content="Based on the provided context, Python was first released in 1991 by Guido van Rossum. It's known for readability and used in web development, data analysis, and AI.")
        else:
             return AIMessage(content="I don't have enough information in the provided context to answer that question accurately.")


llm = MockLLM()
output_parser = StrOutputParser() # To parse the LLM's output to a string

# --- 9. Assemble the RAG chain ---
# This is the core RAG chain:
# Query -> Retrieve -> (Context + Query) -> LLM -> Answer
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | output_parser
)

# --- 10. Run a query ---
query = "When was Python released and who created it? What is it used for?"
print(f"\nUser Query: {query}")
response = rag_chain.invoke(query)
print(f"\nFinal RAG System Response:\n{response}")

# --- Another query (less relevant) ---
print("\n" + "="*50 + "\n")
query_irrelevant = "What is the capital of France?"
print(f"User Query: {query_irrelevant}")
response_irrelevant = rag_chain.invoke(query_irrelevant)
print(f"\nFinal RAG System Response:\n{response_irrelevant}")
