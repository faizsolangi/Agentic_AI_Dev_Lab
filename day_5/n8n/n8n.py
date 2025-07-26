import streamlit as st
import pandas as pd
import requests
import datetime
import os

st.set_page_config(layout="wide", page_title="Vyrant AI - Automation Dashboard")

# --- Configuration ---
# Store these securely in Streamlit Secrets (for local testing, use .streamlit/secrets.toml)
# For Render deployment, configure directly in Render's environment variables for this service.
# Example in .streamlit/secrets.toml:
# N8N_CURATION_WEBHOOK_URL = "https://your-n8n-domain/webhook-test/curation_trigger"
# N8N_DATA_WEBHOOK_URL = "https://your-n8n-domain/webhook-test/data_update" # If agent calls n8n back
# GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit#gid=0"
# (For Google Sheets auth, you'd need service account keys, often stored as secrets too)

N8N_CURATION_WEBHOOK_URL = os.getenv("N8N_CURATION_WEBHOOK_URL") # Use os.getenv for Render env vars
GOOGLE_SHEET_URL = os.getenv("GOOGLE_SHEET_URL") # This would be where your n8n workflow saves data

st.title("Vyrant AI: Agentic Content Curation Dashboard")
st.markdown("---")

# --- Function to load data (Placeholder for real integration) ---
@st.cache_data(ttl=60) # Cache data for 60 seconds
def load_curated_content():
    """
    Placeholder function to load data from Google Sheet or Database.
    In a real application, you'd implement the actual data fetching logic here.
    For demonstration, we'll return dummy data.
    """
    if GOOGLE_SHEET_URL:
        st.info("💡 In a real app, data would be fetched from: " + GOOGLE_SHEET_URL)
        # Example for Google Sheets (requires gspread and setup):
        # import gspread
        # gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        # worksheet = gc.open_by_url(GOOGLE_SHEET_URL).sheet1
        # data = worksheet.get_all_records()
        # df = pd.DataFrame(data)
        # return df

    # Dummy Data for demonstration (replace with actual data source)
    data = {
        "Timestamp": [datetime.datetime.now() - datetime.timedelta(hours=i) for i in range(5)],
        "Title": [
            "Agentic AI Transforms Supply Chains: New Report",
            "Automation Boosts Small Business Productivity",
            "Ethical Considerations in AI Agent Deployment",
            "Future of Work: Human-Agent Collaboration",
            "NOT RELEVANT: New Coffee Shop Opens Downtown"
        ],
        "URL": [
            "https://example.com/ai-supply-chain",
            "https://example.com/smb-automation",
            "https://example.com/ethical-agents",
            "https://example.com/human-agent-future",
            "https://example.com/coffee-news"
        ],
        "Summary": [
            "- AI agents streamline logistics\n- Predictive analytics for inventory\n- Real-time decision making",
            "- Case study on XYZ Inc.\n- Reduced manual tasks by 40%\n- Improved data accuracy",
            "- Importance of transparency\n- Mitigating bias in AI decisions\n- Regulatory compliance",
            "- How agents augment human skills\n- New job roles emerging\n- Collaborative platforms",
            None
        ],
        "Social_Post_Draft": [
            "🚀 Agentic AI is revolutionizing supply chains! Dive into the latest insights on how intelligent automation is boosting efficiency and predicting demand. #AgenticAI #SupplyChain #Automation",
            "SMBs rejoice! Discover how automation can dramatically boost your productivity and cut costs. A must-read for growing businesses. #SMB #Automation #BusinessEfficiency",
            "The rise of AI agents demands ethical thinking. Explore the crucial considerations for responsible AI deployment in business. #EthicalAI #AIDevelopment",
            "The future of work is collaborative! Learn how human-agent partnerships are redefining productivity and creating new opportunities. #FutureOfWork #AIAutomation",
            None
        ],
        "Is_Relevant": [True, True, True, True, False]
    }
    df = pd.DataFrame(data)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(by="Timestamp", ascending=False).reset_index(drop=True)
    return df

# --- UI for Triggering Workflow ---
st.header("Trigger New Content Curation")
trigger_col, status_col = st.columns([1, 2])

with trigger_col:
    if st.button("Manually Run Curation Workflow", help="Triggers the n8n workflow to fetch and process new articles."):
        if N8N_CURATION_WEBHOOK_URL:
            try:
                # You can pass parameters if your n8n webhook expects them, e.g., keywords
                response = requests.post(N8N_CURATION_WEBHOOK_URL, json={"trigger": "manual"})
                if response.status_code == 200:
                    status_col.success("🎉 Agentic workflow triggered successfully!")
                    st.toast("Workflow started!")
                else:
                    status_col.error(f"Failed to trigger n8n workflow. Status: {response.status_code}, Response: {response.text}")
                    st.toast("Workflow trigger failed!")
            except requests.exceptions.RequestException as e:
                status_col.error(f"Network error or n8n not reachable: {e}")
                st.toast("Network error!")
        else:
            status_col.warning("N8N_CURATION_WEBHOOK_URL is not configured. Cannot trigger.")

st.markdown("---")

# --- Display Curated Content ---
st.header("Latest Curated Content & Social Media Drafts")

curated_df = load_curated_content()

if not curated_df.empty:
    st.subheader("All Processed Articles:")
    st.dataframe(curated_df, use_container_width=True)

    st.subheader("Relevant Articles & Drafts:")
    relevant_df = curated_df[curated_df["Is_Relevant"] == True]

    if not relevant_df.empty:
        for index, row in relevant_df.iterrows():
            st.card(
                title=row["Title"],
                content=f"""
                <small>Processed: {row["Timestamp"].strftime("%Y-%m-%d %H:%M")}</small><br>
                **Summary:**<br>{row["Summary"]}<br><br>
                **Social Media Post Draft:**<br>
                `{row["Social_Post_Draft"]}`<br><br>
                [Read Full Article]({row["URL"]})
                """,
                # You can add a button here to copy the post or send for approval
                key=f"article_{index}" # Unique key for each card
            )
            # You could add action buttons if needed here:
            # if st.button(f"Copy Post for '{row['Title']}'", key=f"copy_{index}"):
            #     st.toast("Post copied to clipboard (functionality to be implemented)")
            st.markdown("---")
    else:
        st.info("No relevant articles found yet. Try running the workflow!")
else:
    st.info("No data available to display. Please ensure n8n workflow is running and saving data.")

st.markdown("---")
st.caption("Powered by Vyrant AI - Your Agentic Automation Partner")