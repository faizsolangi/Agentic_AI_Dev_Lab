import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# --- 1. Environment Setup (Same as before) ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY environment variable not set. Please set it in Render.")
    st.stop()

# --- 2. Initialize LLM and Embeddings (using st.cache_resource for efficiency) ---
# Note: These should ideally be initialized only once, outside of functions if possible,
# or passed as args not intended for hashing.
# For simplicity, we'll keep them cached, but be aware that if the model itself changes,
# the cache won't clear based on the model object.
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

# --- 3. Load Data & Create Vector Store (using st.cache_resource for efficiency) ---
@st.cache_resource
# ADD A LEADING UNDERSCORE TO THE UNHASHABLE ARGUMENT
def create_vector_store(text_content, _embeddings_model_obj): # <--- MODIFIED LINE
    st.write("Creating FAISS vector store...")
    text_splitter = CharacterTextSplatter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text_content)
    vector_store = FAISS.from_texts(chunks, _embeddings_model_obj) # <--- USE THE UNDERSCORED ARGUMENT
    st.write("FAISS vector store created successfully.")
    return vector_store

# Define your raw text (could also be loaded from a file uploaded via Streamlit)
raw_text = """
Python is a high-level, interpreted programming language.
It was first released in 1991 by Guido van Rossum.
Python is known for its simplicity and readability, often using indentation to define code blocks.
It supports multiple programming paradigms, including object-oriented, imperative, and functional programming.
Python has a vast standard library and a large, active community that contributes to its extensive ecosystem of third-party libraries.
It is widely used for web development (Django, Flask), data analysis (Pandas, NumPy), artificial intelligence (TensorFlow, PyTorch), scientific computing, and automation.
The Zen of Python is a collection of 19 "guiding principles" for writing computer programs that influence the design of Python.
"""
vector_store = create_vector_store(raw_text, embeddings_model)

# --- 4. Streamlit UI ---
st.set_page_config(page_title="Simple RAG Chatbot", layout="centered")
st.title("🐍 Simple RAG Chatbot (Powered by OpenAI)")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input for user queries
if prompt := st.chat_input("Ask me about Python..."):
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
                retriever=vector_store.as_retriever(),
                return_source_documents=True,
                verbose=True
            )
            response = qa_chain.invoke({"query": prompt})
            result = response["result"]
            source_documents = response["source_documents"]

            st.markdown(result)
            with st.expander("See Source Documents"):
                for doc in source_documents:
                    st.write(f"- **Content:** {doc.page_content[:200]}...")

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": result})