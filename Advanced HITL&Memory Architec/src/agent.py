# my_langgraph_agent01/src/agent.py
import os
import sys

# Ensure the root directory is in Python's path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END, MessagesState
from src.database import get_checkpointer, get_vector_store
# 1. Import the utility function at the top of your file
from utils.graph_utils import save_graph_image
from dotenv import load_dotenv

load_dotenv(".env")
api_key = os.getenv("OPENROUTER_API_KEY")

# 1. Define the custom SDR analytics tool using the @tool decorator
@tool
def analyze_sdr_pipeline(metric_type: str, campaign_name: str = None) -> str:
    """
    Analyzes the SDR pipeline CSV data to compute key metrics like overall reply rate, 
    total meetings booked, or campaign performance summary.
    
    Args:
        metric_type: The type of metric to calculate ('reply_rate', 'total_meetings', 'campaign_summary').
        campaign_name: Optional filter to calculate metrics for a specific campaign name.
    """
    try:
        # Load the SDR pipeline data from the CSV file
        df = pd.read_csv("sdrpipelinedata.csv")

        # Filter by campaign name if provided
        if campaign_name:
            df = df[df['campaign_name'].str.contains(campaign_name, case=False, na=False)]
            # Check if the filtered DataFrame is empty
            if df.empty:
                return f"No records found for campaign: '{campaign_name}'."
        # Calculate metrics based on the requested type
        if metric_type == "reply_rate":
            total_leads = len(df)
            total_replied = df['replied'].sum()
            rate = round((total_replied / total_leads) * 100, 2) if total_leads > 0 else 0.0
            return f"Total Leads: {total_leads}, Replies: {total_replied}, Overall Reply Rate: {rate}%"

        elif metric_type == "total_meetings":
            meetings = df['meeting_booked'].sum()
            return f"Total meetings booked across filtered pipeline: {meetings}"

        elif metric_type == "campaign_summary":
            summary = df.groupby('campaign_name').agg(
                leads=('lead_id', 'count'),
                replies=('replied', 'sum'),
                meetings=('meeting_booked', 'sum')
            ).reset_index().to_dict(orient='records')
            return f"Campaign Performance Summary: {summary}"

        else:
            return f"Unknown metric type '{metric_type}'. Available types: 'reply_rate', 'total_meetings', 'campaign_summary'."

    except FileNotFoundError:
        return "[ERROR] Could not find 'sdrpipelinedata.csv' in the project directory."
    except Exception as e:
        return f"[ERROR] An error occurred: {str(e)}"

# Register all tools into a list
tools = [analyze_sdr_pipeline]

# 2. Initialize Model and bind tools to expose them to Llama 3.1
# using bind_tools(tools) than from langchain.tools import tool, on your LLM model is the modern, standard, and recommended way to handle tool binding in LangChain and LangGraph.
model = ChatOpenAI(
    #model="meta-llama/llama-3.1-8b-instruct",
    model="meta-llama/llama-3.3-70b-instruct",
    api_key=api_key,
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    max_tokens=2000,
# biding the tools to the model allows the model to recognize and utilize the tools during its reasoning process. This is crucial for enabling the model to call external functions or APIs as part of its response generation.
).bind_tools(tools)

def call_model(state: MessagesState, config, *, store):
    """Core agent node that queries pgvector memory and prompts the model."""
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    namespace = ("users", user_id)
    latest_message = state["messages"][-1].content
    
    # Semantic search via pgvector
    memories = store.search(namespace, query=latest_message, limit=3)
    memory_context = "\n".join([f"- {m.value.get('text')}" for m in memories]) if memories else "No prior memories found."

    system_prompt = (
        "You are an expert SDR Analytics AI assistant with long-term memory and data tools. "
        "Use retrieved user memories when relevant, and call your tools whenever pipeline data analysis is requested:\n"
        f"{memory_context}"
    )
    # Build the message list for the model invocation
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    # Invoke the model with the constructed messages
    response = model.invoke(messages)
    # The model's response is returned as a dictionary containing the messages, which can be further processed or stored as needed.
    return {"messages": [response]}

# 3. Build the workflow graph with conditional tool routing
workflow = StateGraph(MessagesState)

# Add nodes for the agent and the execution of tools
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

# Define execution edges
workflow.add_edge(START, "agent")

# Conditional edge: checks if model called a tool. If yes -> "tools" node, if no -> END
workflow.add_conditional_edges("agent", tools_condition)

# Crucial feedback loop: once tools finish executing, return control back to the agent
workflow.add_edge("tools", "agent")

checkpointer = get_checkpointer()
store = get_vector_store()

graph = workflow.compile(checkpointer=checkpointer, store=store)

# 2. Call it cleanly in your execution block
if __name__ == "__main__":
    # Call the external utility function, passing the compiled graph
    save_graph_image(graph, output_filename="sdr_agent_architecture.png")

    # Test invoking the graph directly
    print("\n--- Testing Agent Invocation ---")
    config = {"configurable": {"user_id": "sergio_user", "thread_id": "test-1"}}
    response = graph.invoke({"messages": [{"role": "user", "content": "Hello!"}]}, config=config)
    print("Agent Response:", response["messages"][-1].content)

