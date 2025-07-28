import requests
import smtplib
from email.mime.text import MIMEText
from notion_client import Client
from langchain import LLMChain, PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from crewai import Agent, Task, Crew
import asyncio
import platform
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Notion client
notion = Client(auth=os.getenv("NOTION_API_KEY"))
database_id = os.getenv("NOTION_DATABASE_ID")

# SMTP configuration for Google Workspace
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = os.getenv("SMTP_USER", "outreach@solinnovate.io")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

# Apollo.io API for scraping coaching leads
def scrape_leads(api_key, industry="Coaching", limit=50):
    url = "https://api.apollo.io/v1/people"
    params = {"api_key": api_key, "industry": industry, "per_page": limit}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Apollo API error: {response.status_code} - {response.text}")
    leads = response.json().get("people", [])
    for lead in leads:
        notion.pages.create(
            parent={"database_id": database_id},
            properties={
                "Name": {"title": [{"text": {"content": lead.get("name", "Unknown")}}]},
                "Email": {"email": lead.get("email", "")},
                "Industry": {"select": {"name": lead.get("industry", "Coaching")}},
                "Status": {"select": {"name": "New"}},  # Manual tracking
                "Score": {"number": 0},
                "Message": {"rich_text": [{"text": {"content": ""}}]}
            }
        )
    return leads

# LangChain for generating coaching-specific emails
def generate_email(name, industry):
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))
    template = """
    Subject: Free 15-Min Coaching Workflow Audit

    Hi {name},
    As a coach in the {industry} space, you could save hours weekly with AI automation.
    I’d love to offer you a free 15-min audit to optimize your processes.
    Best,
    [Your Name]
    solinnovate.io
    """
    prompt = PromptTemplate(template=template, input_variables=["name", "industry"])
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(name=name, industry=industry)

# Send email via Google Workspace SMTP
def send_email(to_email, email_content):
    msg = MIMEText(email_content)
    msg["Subject"] = "Free 15-Min Coaching Workflow Audit"
    msg["From"] = SMTP_USER
    msg["To"] = to_email

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, to_email, msg.as_string())
        print(f"Email sent successfully to {to_email}")
    except Exception as e:
        print(f"Failed to send email to {to_email}: {e}")

# CrewAI for lead scoring
def score_leads(leads):
    scorer = Agent(
        role="Lead Scorer",
        goal="Score coaching leads based on industry fit",
        backstory="Expert in evaluating coaching professionals",
        llm=ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))
    )
    task = Task(
        description="Score leads based on industry (e.g., +10 for Coaching, +5 for Training)",
        agent=scorer
    )
    crew = Crew(agents=[scorer], tasks=[task])
    crew.kickoff()
    scored_leads = []
    for lead in leads:
        industry = lead.get("industry", "Coaching")
        score = 10 if industry == "Coaching" else 5 if "Training" in industry else 0
        scored_leads.append({"name": lead.get("name", "Unknown"), "score": score})
    return scored_leads

# Streamlit dashboard
def run_dashboard():
    st.title("Coaching Outreach Bot Dashboard")
    leads = notion.databases.query(database_id=database_id)["results"]
    st.write("### Leads Overview")
    for lead in leads:
        name = lead["properties"]["Name"]["title"][0]["text"]["content"]
        email = lead["properties"]["Email"]["email"]
        industry = lead["properties"]["Industry"]["select"]["name"]
        status = lead["properties"]["Status"]["select"]["name"]
        score = lead["properties"]["Score"]["number"]
        st.write(f"Name: {name}, Email: {email}, Industry: {industry}, Status: {status}, Score: {score}")
    if st.button("Refresh Scores"):
        api_key = os.getenv("APOLLO_API_KEY")
        leads = scrape_leads(api_key)
        scored_leads = score_leads(leads)
        for lead, scored in zip(leads, scored_leads):
            st.write(f"Scored {scored['name']}: {scored['score']}")
        st.experimental_rerun()

# Main function
async def main():
    api_key = os.getenv("APOLLO_API_KEY")
    # Step 1: Scrape coaching leads
    leads = scrape_leads(api_key, industry="Coaching")
    # Step 2: Generate and send emails
    for lead in leads:
        name = lead.get("name", "Unknown")
        email = lead.get("email", "")
        if email:
            email_content = generate_email(name, "Coaching")
            send_email(email, email_content)
    # Step 3: Run dashboard
    run_dashboard()

# Pyodide-compatible main loop
if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
