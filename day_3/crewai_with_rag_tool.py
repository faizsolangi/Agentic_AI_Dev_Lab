import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from crewai_tools import SerperDevTool, tool # NEW IMPORT for custom tool decorator
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

# --- Load environment variables ---
# For local development, uncomment if you use a .env file
# from dotenv import load_dotenv
# load_dotenv()

# --- 1. Set up your API Keys ---
# Ensure OPENAI_API_KEY and SERPER_API_KEY are set in your environment variables.
# For local testing, remember to export them in your terminal:
# export OPENAI_API_KEY='sk-YOUR_OPENAI_API_KEY'
# export SERPER_API_KEY='your_serper_api_key'

# --- 2. Initialize LLM and Embeddings for Agents & RAG ---
llm = ChatOpenAI(
    model="gpt-3.5-turbo", # Consistent LLM for agents and RAG
    temperature=0.7,
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

# --- 3. RAG System Setup (Pre-build the Vector Store) ---
# This RAG knowledge base will be about Python, same as Day 2
default_raw_text = """
Python is a high-level, interpreted programming language.
It was first released in 1991 by Guido van Rossum.
Python is known for its simplicity and readability, often using indentation to define code blocks.
It supports multiple programming paradigms, including object-oriented, imperative, and functional programming.
Python has a vast standard library and a large, active community that contributes to its extensive ecosystem of third-party libraries.
It is widely used for web development (Django, Flask), data analysis (Pandas, NumPy), artificial intelligence (TensorFlow, PyTorch), scientific computing, and automation.
The Zen of Python is a collection of 19 "guiding principles" for writing computer programs that influence the design of Python.
"""

# Process the text into LangChain Document objects
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
rag_documents = text_splitter.create_documents([default_raw_text])

# Create the FAISS vector store once
print("Building RAG Vector Store...")
rag_vector_store = FAISS.from_documents(rag_documents, embeddings_model)
print("RAG Vector Store built successfully.")

# --- 4. Define Custom RAG Tool for CrewAI ---
# This function will be called by the agent
@tool("RAG Python Info Tool") # Use the @tool decorator to define the tool
def rag_python_info_tool(query: str) -> str:
    """
    A tool to retrieve specific information about Python from a pre-built knowledge base.
    Use this for direct questions about Python's history, features, or applications.
    Input should be a clear question about Python.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=rag_vector_store.as_retriever(), # Use the pre-built vector store
        return_source_documents=True
    )
    result = qa_chain.invoke({"query": query})
    # Return only the relevant answer part, as the agent needs a concise tool output
    return result["result"]

# --- 5. Define Other Tools ---
search_tool = SerperDevTool() # Web search tool for general, real-time info

# --- 6. Define Agents ---
researcher = Agent(
    role='Senior AI & Python Researcher',
    goal='Find and synthesize comprehensive information on AI trends and specific Python knowledge.',
    backstory="""You are a highly analytical and meticulous researcher, expert in both
    cutting-edge AI developments and the intricate details of the Python programming
    language. You efficiently use specialized tools to gather precise information.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[search_tool, rag_python_info_tool] # <--- NEW: Assign both tools
)

writer = Agent(
    role='Professional Article Writer',
    goal='Write an engaging and informative short article (around 300 words) based on research findings',
    backstory="""You are a renowned writer known for transforming complex technical
    information into clear, compelling, and accessible articles. You excel at
    structuring content and maintaining an engaging tone.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# --- 7. Define Tasks ---
research_task = Task(
    description="""Conduct a thorough investigation into the current trends in AI,
    including but not limited to Generative AI, Responsible AI, AI in healthcare,
    and Edge AI, using the web search tool for up-to-date information.
    Additionally, answer the following question about Python: 'Who created Python and when?'
    You MUST use the 'RAG Python Info Tool' to answer the Python question.
    Your final answer MUST be a comprehensive summary of these findings,
    presented as bullet points, clearly separating AI trends from the Python answer,
    and citing any sources briefly.""",
    expected_output="A bullet-point summary of current AI trends (from web search) AND the specific answer to 'Who created Python and when?' (from RAG tool), suitable for an article.",
    agent=researcher
)

write_task = Task(
    description="""Write a short, engaging, and informative article (around 300 words)
    about the current state of Artificial Intelligence and its connection to Python's role in AI,
    incorporating the trends and challenges provided by the researcher, as well as the specific
    Python information retrieved.
    Your article should have an introduction, main body paragraphs covering AI trends,
    Python's role, and a concluding thought.
    Your final answer MUST be the complete article, formatted professionally.""",
    expected_output="A well-structured 300-word article on current AI trends and Python's role in AI.",
    agent=writer,
    context=[research_task]
)

# --- 8. Form the Crew ---
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True
)

# --- 9. Kick off the Crew's Work ---
print("## Crew starting its work with RAG and Web Search tools...")
result = crew.kickoff()
print("\n## Crew work finished!")
print("\n### Final Article:")
print(result)