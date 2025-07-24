import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

# --- 1. Environment Setup ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY environment variable not set. Please set it in Render.")
    st.stop()

# --- 2. Initialize LLM and Embeddings ---
@st.cache_resource
def initialize_models():
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=OPENAI_API_KEY
    )
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    return llm, embeddings_model

llm, embeddings_model = initialize_models()

# --- 3. Function to load and split documents from various sources ---
def load_and_split_documents(text_content=None, url=None):
    all_documents = []

    # Initialize text splitter for both local and web content
    common_text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    if text_content:
        # Load from provided string, ensure it returns Document objects
        # MODIFIED LINE: Use create_documents instead of split_text
        text_chunks = common_text_splitter.create_documents([text_content])
        all_documents.extend(text_chunks)
        st.info(f"Loaded {len(text_chunks)} chunks from provided text.")

    if url:
        try:
            # Load from URL
            st.info(f"Loading content from URL: {url}...")
            web_loader = WebBaseLoader(url)
            web_docs = web_loader.load()

            # Split web documents, which are already Document objects
            web_chunks = common_text_splitter.split_documents(web_docs)
            all_documents.extend(web_chunks)
            st.success(f"Successfully loaded {len(web_chunks)} chunks from {url}.")
        except Exception as e:
            st.error(f"Error loading content from URL {url}: {e}")

    return all_documents

# --- 4. Create Vector Store (Refactored to accept documents directly) ---
@st.cache_resource(hash_funcs={OpenAIEmbeddings: lambda _: None})
# MODIFIED LINE: Add underscore to 'documents'
def create_vector_store_from_documents(_embeddings_model_obj, _documents):
    if not _documents: # Use the underscored name here
        st.warning("No documents to create vector store from.")
        return None
    st.write("Creating FAISS vector store...")
    # Use the underscored name here
    vector_store = FAISS.from_documents(_documents, _embeddings_model_obj)
    st.write("FAISS vector store created successfully.")
    return vector_store


# --- 5. Streamlit UI (Rest of the code remains the same) ---
st.set_page_config(page_title="Dynamic RAG Chatbot", layout="centered")
st.title("🌐 Dynamic RAG Chatbot (Powered by OpenAI)")

st.subheader("Knowledge Base Configuration")

# --- Default Raw Text ---
default_raw_text = """
Python is a high-level, interpreted programming language.
It was first released in 1991 by Guido van Rossum.
Python is known for its simplicity and readability, often using indentation to define code blocks.
It supports multiple programming paradigms, including object-oriented, imperative, and functional programming.
Python has a vast standard library and a large, active community that contributes to its extensive ecosystem of third-party libraries.
It is widely used for web development (Django, Flask), data analysis (Pandas, NumPy), artificial intelligence (TensorFlow, PyTorch), scientific computing, and automation.
The Zen of Python is a collection of 19 "guiding principles" for writing computer programs that influence the design of Python.
"""

st.write("Using default information about Python.")

# --- User-provided URL input ---
url_input = st.text_input("Or enter a URL to add to the knowledge base (e.g., https://www.example.com):", key="url_input")

if st.button("Load/Re-index Knowledge Base"):
    with st.spinner("Loading new data and building vector store..."):
        st.session_state['current_documents'] = load_and_split_documents(
            text_content=default_raw_text, # Keep default text
            url=url_input # Add content from URL
        )
        st.session_state['vector_store'] = create_vector_store_from_documents(embeddings_model, st.session_state['current_documents'])
        st.success("Knowledge base updated!")

# Initialize vector store on first run or if not yet in session_state
if 'vector_store' not in st.session_state or st.session_state['vector_store'] is None:
    st.session_state['current_documents'] = load_and_split_documents(text_content=default_raw_text)
    # The call to the function remains the same, as the underscore only affects the *definition* of the cached function's arguments.
    st.session_state['vector_store'] = create_vector_store_from_documents(embeddings_model, st.session_state['current_documents'])




st.subheader("Chat with the RAG System")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input for user queries
if prompt := st.chat_input("Ask me about the loaded content..."):
    if st.session_state['vector_store'] is None:
        st.warning("Please load the knowledge base first.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get RAG response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state['vector_store'].as_retriever(),
                    return_source_documents=True,
                    verbose=True
                )
                response = qa_chain.invoke({"query": prompt})
                result = response["result"]
                source_documents = response["source_documents"]

                st.markdown(result)
                with st.expander("See Source Documents"):
                    for doc in source_documents:
                        source_info = doc.metadata.get('source', 'N/A')
                        st.write(f"- **Source:** {source_info}")
                        st.write(f"  **Content:** {doc.page_content[:200]}...")

            st.session_state.messages.append({"role": "assistant", "content": result})