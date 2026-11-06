import os
import sys
import pandas as pd

# Ensure the project root directory is in Python's path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
from src.agent import graph

load_dotenv(".env")

def create_sample_csv_if_missing():
    """Helper function to create a mock sdrpipelinedata.csv file in the project root if missing."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    csv_path = os.path.join(root_dir, "sdrpipelinedata.csv")
    
    if not os.path.exists(csv_path):
        print("\n[Setup] 'sdrpipelinedata.csv' not found. Creating a sample dataset for testing...")
        data = {
            "lead_id": [1, 2, 3, 4, 5, 6],
            "company_name": ["Acme Corp", "Beta LLC", "Gamma Inc", "Delta Co", "Echo Ltd", "Zeta Tech"],
            "campaign_name": ["Q4_Enterprise", "Q4_Enterprise", "SMB_Outbound", "SMB_Outbound", "Q4_Enterprise", "SMB_Outbound"],
            "status": ["Replied", "Contacted", "Meeting Booked", "Contacted", "Replied", "Meeting Booked"],
            "emails_sent": [2, 3, 1, 4, 2, 2],
            "replied": [1, 0, 1, 0, 1, 1],
            "meeting_booked": [0, 0, 1, 0, 0, 1],
            "date_contacted": ["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05", "2026-10-06"]
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print("✓ Sample CSV created successfully at root.")

def test_sdr_tool_execution():
    print("\n========================================")
    print("🛠️ TESTING SDR TOOL-CALLING & EXECUTION")
    print("========================================")
    
    # Ensure data file is available
    create_sample_csv_if_missing()
    
    user_id = "sergio_user"
    thread_id = "tool_test_thread_01"
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    
    # Ask a question that requires calling the Pandas analytics tool
    query = "Can you calculate our overall reply rate and give me a campaign summary?"
    print(f"\nUser > {query}")
    
    messages = {"messages": [("user", query)]}
    
    # Invoke the graph: LangGraph will parse the intent, execute the tool, and feed results back to the model
    response = graph.invoke(messages, config=config)
    
    print(f"\nAgent > {response['messages'][-1].content}")
    print("\n========================================")
    print("✨ TOOL EXECUTION TEST COMPLETED")
    print("========================================")

if __name__ == "__main__":
    test_sdr_tool_execution()