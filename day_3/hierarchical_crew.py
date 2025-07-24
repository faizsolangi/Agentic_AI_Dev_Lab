import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai_tools import SerperDevTool

# --- Load environment variables ---
# For local development, uncomment if you use a .env file
# from dotenv import load_dotenv
# load_dotenv()

# --- 1. Set up your API Keys ---
# Ensure OPENAI_API_KEY and SERPER_API_KEY are set in your environment variables.
# export OPENAI_API_KEY='sk-YOUR_OPENAI_API_KEY'
# export SERPER_API_KEY='your_serper_api_key'

# --- 2. Initialize LLM for Agents ---
llm = ChatOpenAI(
    model="gpt-4o", # Using a more capable model for hierarchical processes is often recommended
    temperature=0.7,
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

# --- 3. Define Tools (if any, typically for specialized agents) ---
search_tool = SerperDevTool()

# --- 4. Define Agents ---

# The Manager Agent
# This agent will orchestrate the entire process, delegate tasks, and ensure goals are met.
# It MUST have allow_delegation=True.
manager = Agent(
    role='Marketing Project Manager',
    goal='Oversee the creation of a comprehensive marketing brief for a new product launch, ensuring all aspects are covered and aligned with marketing goals.',
    backstory="""You are an experienced Marketing Project Manager with a keen eye for detail and
    a knack for leading cross-functional teams. You excel at breaking down complex
    marketing initiatives into manageable tasks and ensuring high-quality deliverables.
    You know how to delegate effectively and review work to maintain standards.""",
    verbose=True,
    llm=llm,
    allow_delegation=True # <--- Crucial for a manager agent
)

# The Content Strategist Agent
# This agent will handle the research and strategy for the marketing brief.
content_strategist = Agent(
    role='AI Product Content Strategist',
    goal='Develop a content strategy for a new AI-powered product, including target audience analysis, key messaging, and competitive landscape research.',
    backstory="""You are a brilliant Content Strategist specializing in cutting-edge AI products.
    You deeply understand market trends, audience needs, and how to craft compelling
    narratives that resonate. You are meticulous in your research and strategic planning.""",
    verbose=True,
    llm=llm,
    tools=[search_tool], # This agent needs a tool for research
    allow_delegation=False # Typically, worker agents don't delegate further unless designed to
)

# The Copywriter Agent
# This agent will focus on crafting the actual copy based on the strategy.
copywriter = Agent(
    role='Creative AI Copywriter',
    goal='Write engaging and persuasive marketing copy based on strategic briefs and research findings for AI-powered products.',
    backstory="""You are a highly creative and skilled copywriter known for transforming
    complex technical concepts into clear, engaging, and persuasive marketing messages.
    You excel at short-form and long-form content, always with the target audience in mind.""",
    verbose=True,
    llm=llm,
    allow_delegation=False
)

# --- 5. Define the Single Top-Level Task for the Manager ---
# In a hierarchical process, you usually define ONE main task for the manager.
# The manager will then internally break it down and delegate.
main_marketing_task = Task(
    description="""Create a comprehensive marketing brief for a new AI-powered Personal Assistant product.
    The brief should include:
    1.  Target Audience Demographics and Psychographics.
    2.  Key Features and Benefits of the AI Assistant.
    3.  Unique Selling Propositions (USPs) compared to competitors.
    4.  Recommended Marketing Channels.
    5.  A draft of a compelling marketing headline and tagline.

    Ensure the brief is well-researched and actionable for a marketing team.""",
    expected_output="A detailed, actionable marketing brief document in markdown format, covering all the requested sections.",
    agent=manager, # This task is assigned to the manager
)

# --- 6. Form the Crew with Hierarchical Process ---
crew = Crew(
    agents=[manager, content_strategist, copywriter], # List all agents in the crew
    tasks=[main_marketing_task], # Only the top-level task is provided here
    process=Process.hierarchical, # <--- Set the process to hierarchical
    manager_llm=llm, # <--- Crucial: Specify which LLM the manager agent uses for its reasoning/delegation
    verbose=True
)

# --- 7. Kick off the Crew's Work ---
print("## Crew starting its work with Hierarchical Process...")
result = crew.kickoff()
print("\n## Crew work finished!")
print("\n### Final Marketing Brief:")
print(result)