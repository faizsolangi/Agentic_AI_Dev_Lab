import requests
import smtplib
from email.mime.text import MIMEText
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from crewai import Agent, Task, Crew
import os
from dotenv import load_dotenv
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Load environment variables
load_dotenv()

# Google Sheets setup
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
private_key = os.getenv("GOOGLE_PRIVATE_KEY")
if "\\n" in private_key:
    private_key = private_key.replace("\\n", "\n")
creds_data = {
    "type": os.getenv("GOOGLE_TYPE", "service_account"),
    "project_id": os.getenv("GOOGLE_PROJECT_ID"),
    "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
    "private_key": private_key,
    "client_email": os.getenv("GOOGLE_CLIENT_EMAIL"),
    "client_id": os.getenv("GOOGLE_CLIENT_ID"),
    "auth_uri": os.getenv("GOOGLE_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
    "token_uri": os.getenv("GOOGLE_TOKEN_URI", "https://oauth2.googleapis.com/token"),
    "auth_provider_x509_cert_url": os.getenv("GOOGLE_AUTH_PROVIDER_X509_CERT_URL", "https://www.googleapis.com/oauth2/v1/certs"),
    "client_x509_cert_url": os.getenv("GOOGLE_CLIENT_X509_CERT_URL", "https://www.googleapis.com/robot/v1/metadata/x509/outreachbottracker%40coaching-leads-tracker.iam.gserviceaccount.com")
}
for key, value in creds_data.items():
    if not value:
        raise ValueError(f"Missing or empty environment variable: {key}")
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_data, scope)
client = gspread.authorize(creds)
spreadsheet_id = os.getenv("GOOGLE_SPREADSHEET_ID")
if not spreadsheet_id:
    raise ValueError("GOOGLE_SPREADSHEET_ID environment variable not set")
sheet = client.open_by_key(spreadsheet_id).sheet1

# Apollo.io API for searching leads




def scrape_leads(industries=None, search_terms=None):
    st.write("Starting scrape_leads function")
    print("Starting scrape_leads function")  # For Render logs
    api_key = os.getenv("APOLLO_API_KEY")
    if not api_key:
        st.write("Error: APOLLO_API_KEY not found in environment")
        print("Error: APOLLO_API_KEY not found in environment")
        return []
    st.write(f"API key (partial): {api_key[:5]}...{api_key[-5:]}")
    print(f"API key (partial): {api_key[:5]}...{api_key[-5:]}")
    url = "https://api.apollo.io/api/v1/api/v1/accounts/search"  # Test alternative endpoint
    headers = {
        "accept": "application/json",
        "Cache-Control": "no-cache",
        "Content-Type": "application/json",
        "X-Api-Key": api_key
    }
    payload = {
        "per_page": 10,
        "page": 1
    }
    st.write(f"Sending request with payload: {payload}")
    print(f"Sending request with payload: {payload}")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        st.write(f"Response status: {response.status_code}")
        print(f"Response status: {response.status_code}")
        if response.status_code != 200:
            st.write(f"Apollo API error: {response.status_code} - {response.text}")
            print(f"Apollo API error: {response.status_code} - {response.text}")
            return []
        response_data = response.json()
        st.write(f"Full response: {response_data}")
        print(f"Full response: {response_data}")
        leads = response_data.get("organizations", [])  # Adjusted for organizations
        st.write(f"Number of organizations retrieved: {len(leads)}")
        print(f"Number of organizations retrieved: {len(leads)}")
        for lead in leads:
            sheet.append_row([
                lead.get("name", "Unknown"),
                "",
                lead.get("industry", "Unknown"),
                "New",
                0,
                ""
            ])
        return leads
    except Exception as e:
        st.write(f"Exception in scrape_leads: {str(e)}")
        print(f"Exception in scrape_leads: {str(e)}")
    st.write("Finished scrape_leads function")
    print("Finished scrape_leads function")
    return []













# LangChain for generating emails
def generate_email(name, industry):
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))
    template = """
    Subject: Free 15-Min Workflow Audit

    Hi {name},
    As a professional in the {industry} space, you could save hours weekly with AI automation.
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
    msg["Subject"] = "Free 15-Min Workflow Audit"
    msg["From"] = os.getenv("SMTP_USER", "outreach@solinnovate.io")
    msg["To"] = to_email
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(os.getenv("SMTP_USER", "outreach@solinnovate.io"), os.getenv("SMTP_PASSWORD"))
        server.sendmail(os.getenv("SMTP_USER", "outreach@solinnovate.io"), to_email, msg.as_string())
    print(f"Email sent successfully to {to_email}")  # For Render logs

# CrewAI for lead scoring
def score_leads(leads, industries=None):
    scorer = Agent(
        role="Lead Scorer",
        goal="Score leads based on Apollo data and selected industries",
        backstory="Expert in evaluating professionals across multiple industries",
        llm=ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))
    )
    industries = industries or ["Technology", "Healthcare"]
    task = Task(
        description=f"Score leads based on industry and title: +10 for titles containing industry-specific keywords ('{', '.join([ind.lower().split()[0] for ind in industries])}' for {', '.join(industries)}), +5 for 'Training' or 'Consultant' roles, +2 for leads in the selected industries {', '.join(industries)}",
        expected_output="A list of dictionaries containing 'name', 'score', and 'email' for each lead",
        agent=scorer
    )
    crew = Crew(agents=[scorer], tasks=[task])
    crew.kickoff()
    scored_leads = []
    for lead in leads:
        industry = lead.get("organization_industry", "").lower()
        title = lead.get("job_title", "").lower()
        base_score = 2 if any(ind.lower() in industry for ind in industries) else 0
        title_score = 10 if any(keyword in title for keyword in [ind.lower().split()[0] for ind in industries]) else 5 if "training" in title or "consultant" in title else 0
        total_score = base_score + title_score
        scored_leads.append({"name": lead.get("name", "Unknown"), "score": total_score, "email": lead.get("email", "")})
    return scored_leads

# Streamlit dashboard
def run_dashboard():
    st.title("Multi-Industry Outreach Bot Demo")
    industries = st.multiselect("Select Industries", ["Technology", "Healthcare", "Coaching", "Education"], default=["Technology", "Healthcare"])
    search_terms = st.text_input("Enter Search Terms (comma-separated)", value="Manager,Lead").split(",")
    search_terms = [term.strip() for term in search_terms if term.strip()]
    if "leads" not in st.session_state:
        try:
            st.session_state.leads = scrape_leads(industries, search_terms)
            st.write("Initial leads fetched:", len(st.session_state.leads))
        except Exception as e:
            st.write(f"Error during initial fetch: {str(e)}")
    if st.button("Run Search"):
        try:
            st.session_state.leads = scrape_leads(industries, search_terms)
            st.write("Leads after search:", len(st.session_state.leads))
        except Exception as e:
            st.write(f"Error during search: {str(e)}")
        st.rerun()
    leads = st.session_state.leads
    st.write("### Leads Overview")
    if not leads:
        st.write("No leads found. Check UI messages and Render logs for details.")
    for i, lead in enumerate(leads):
        name = lead.get("name", "Unknown")
        email = lead.get("email", "")
        industry = lead.get("organization_industry", "")
        score = next((s["score"] for s in score_leads([lead], industries) if s["name"] == name), 0)
        status = sheet.row_values(i + 2)[3] if i + 2 <= len(sheet.get_all_values()) else "New"
        st.write(f"Name: {name}, Email: {email}, Industry: {industry}, Status: {status}, Score: {score}")
    if st.button("Refresh Leads and Send Emails"):
        try:
            st.session_state.leads = scrape_leads(industries, search_terms)
            scored_leads = score_leads(st.session_state.leads, industries)
            for lead in scored_leads:
                if lead["email"]:
                    email_content = generate_email(lead["name"], lead.get("organization_industry", "Technology & Healthcare"))
                    send_email(lead["email"], email_content)
        except Exception as e:
            st.write(f"Error during refresh: {str(e)}")
        st.rerun()

# Render web service entry point
import sys
import waitress

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        waitress.serve(run_dashboard, host="0.0.0.0", port=8501)
    else:
        run_dashboard()
