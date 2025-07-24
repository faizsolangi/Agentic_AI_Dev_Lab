import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI # We'll use your existing OpenAI setup

# --- 1. Set up your OpenAI API Key (already done, but re-confirming) ---
# Ensure OPENAI_API_KEY is set in your environment variables on Render
# For local testing, you can export it in your terminal:
# export OPENAI_API_KEY='sk-YOUR_OPENAI_API_KEY'

# Initialize the OpenAI LLM
# We're using gpt-3.5-turbo for a cost-effective start.
# If you want to use a different model or provider, you'd configure it here.
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    # If OPENAI_API_KEY is in env vars, langchain_openai often picks it up automatically.
    # But explicitly passing it can be good practice.
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

# --- 2. Define Agents ---
# Agent 1: Researcher
researcher = Agent(
    role='Senior Researcher',
    goal='Find and summarize the latest trends and challenges in Artificial Intelligence',
    backstory="""You are a seasoned AI researcher with a knack for identifying
    cutting-edge developments and critical challenges in the field.
    Your summaries are always concise, insightful, and accurate.""",
    verbose=True, # See the agent's thought process
    allow_delegation=False, # For simplicity, agents won't delegate in this first example
    llm=llm # Assign the OpenAI LLM to this agent
)

# Agent 2: Writer
writer = Agent(
    role='Professional Article Writer',
    goal='Write an engaging and informative short article (around 300 words) based on research findings',
    backstory="""You are a renowned writer known for transforming complex technical
    information into clear, compelling, and accessible articles. You excel at
    structuring content and maintaining an engaging tone.""",
    verbose=True, # See the agent's thought process
    allow_delegation=False,
    llm=llm # Assign the OpenAI LLM to this agent
)

# --- 3. Define Tasks ---
# Task for the Researcher
research_task = Task(
    description="""Conduct a thorough investigation into the current trends in AI,
    including but not limited to Generative AI, Responsible AI, AI in healthcare,
    and Edge AI. Also identify key challenges such as ethical considerations,
    data privacy, and scalability.
    Your final answer MUST be a comprehensive summary of these findings,
    presented as bullet points.""",
    expected_output="A bullet-point summary of current AI trends and challenges.",
    agent=researcher # Assign this task to the researcher agent
)

# Task for the Writer
write_task = Task(
    description="""Write a short, engaging, and informative article (around 300 words)
    about the current state of Artificial Intelligence, incorporating the trends
    and challenges provided by the researcher.
    Your article should have an introduction, main body paragraphs covering trends
    and challenges, and a concluding thought.
    Your final answer MUST be the complete article, formatted professionally.""",
    expected_output="A well-structured 300-word article on current AI trends and challenges.",
    agent=writer, # Assign this task to the writer agent
    context=[research_task] # The writer's task depends on the output of the research_task
)

# --- 4. Form the Crew ---
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential, # Tasks will be executed one after another
    verbose=True # See the crew's overall execution flow
)

# --- 5. Kick off the Crew's Work ---
print("## Crew starting its work...")
result = crew.kickoff()
print("\n## Crew work finished!")
print("\n### Final Article:")
print(result)