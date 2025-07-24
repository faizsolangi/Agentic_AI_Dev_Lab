import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai_tools import SerperDevTool # NEW IMPORT for web search tool

# --- Load environment variables ---
# This line is primarily for local development. Render handles env vars automatically.
# from dotenv import load_dotenv
# load_dotenv()

# --- 1. Set up your API Keys ---
# Ensure OPENAI_API_KEY and SERPER_API_KEY are set in your environment variables.
# For local testing, remember to export them in your terminal:
# export OPENAI_API_KEY='sk-YOUR_OPENAI_API_KEY'
# export SERPER_API_KEY='your_serper_api_key'

# Initialize the OpenAI LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo", # You can try "gpt-4" or "gpt-4o" for better reasoning, but higher cost.
    temperature=0.7,
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

# --- Define the Web Search Tool ---
# This tool uses the SERPER_API_KEY from your environment variables.
search_tool = SerperDevTool()

# --- 2. Define Agents ---
# Agent 1: Researcher (now with a tool)
researcher = Agent(
    role='Senior Researcher',
    goal='Find and summarize the latest trends and challenges in Artificial Intelligence, focusing on real-world impact and future predictions.',
    backstory="""You are a seasoned AI researcher with a knack for identifying
    cutting-edge developments and critical challenges in the field. You are
    diligent, thorough, and leverage powerful search tools to get the most
    up-to-date information. Your summaries are always concise, insightful, and accurate.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[search_tool] # <--- NEW: Assign the search tool to the researcher
)

# Agent 2: Writer (remains the same)
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

# --- 3. Define Tasks ---
# Task for the Researcher (description updated to encourage tool use)
research_task = Task(
    description="""Conduct a thorough investigation into the current trends in AI,
    including but not limited to Generative AI, Responsible AI, AI in healthcare,
    and Edge AI. Use your search tool to find recent articles, news, and reports
    (post-2023). Also identify key challenges such as ethical considerations,
    data privacy, and scalability.
    Your final answer MUST be a comprehensive summary of these findings,
    presented as bullet points, citing any sources briefly if relevant.""",
    expected_output="A bullet-point summary of current AI trends and challenges, with emphasis on recent developments and challenges, suitable for an article.",
    agent=researcher
)

# Task for the Writer (remains the same, but now the context comes from tool-enhanced research)
write_task = Task(
    description="""Write a short, engaging, and informative article (around 300 words)
    about the current state of Artificial Intelligence, incorporating the trends
    and challenges provided by the researcher.
    Your article should have an introduction, main body paragraphs covering trends
    and challenges, and a concluding thought.
    Your final answer MUST be the complete article, formatted professionally.""",
    expected_output="A well-structured 300-word article on current AI trends and challenges.",
    agent=writer,
    context=[research_task]
)

# --- 4. Form the Crew ---
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True
)

# --- 5. Kick off the Crew's Work ---
print("## Crew starting its work with tools...")
result = crew.kickoff()
print("\n## Crew work finished!")
print("\n### Final Article:")
print(result)