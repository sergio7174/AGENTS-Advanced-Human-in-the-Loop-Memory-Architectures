# Advanced HITL&Memory Architec/src/M3_memory_summarization.py

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from typing import TypedDict, Annotated, List
import operator

load_dotenv()

# 1. Define Extended State with a Summary Field
class State(TypedDict):
    # Annotated is used here to provide additional metadata about the 'messages' field, indicating that it is a list of messages and that the operator.add function can be used to combine lists of messages. This allows for more flexible handling of message history within the state, enabling the system to accumulate messages over time while also maintaining a long-term semantic summary of the conversation. 
    messages: Annotated[List, operator.add]
    summary: str  # Long-term semantic summary of older conversation turns

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# 2. Node to Process User Input and Respond
def call_model(state: State):
    summary = state.get("summary", "")
    messages = state["messages"]
    
    # Inject long-term summary into context if it exists
    system_prompt = "You are a helpful AI assistant."
    if summary:
        system_prompt += f"\n\n[Background Memory / Summary of Past Context]:\n{summary}"
        
    response = model.invoke([SystemMessage(content=system_prompt)] + messages)
    return {"messages": [response]}

# 3. Reflection & Semantic Summarization Node
def summarize_conversation(state: State):
    """Reflects on the message history and updates the semantic summary."""
    summary = state.get("summary", "")
    messages = state["messages"]
    
    # Create prompt for the summarizer LLM
    print("\n🧠 [Reflection Triggered] Synthesizing conversation history into long-term summary...")
    
    summarization_prompt = (
        "This is a summary of the conversation so far: {summary}\n\n"
        "Extend and update this summary using the new messages below, capturing key facts, user preferences, "
        "and decisions made. Keep it concise and semantic."
    )
    # summarization_prompt.format is used to dynamically insert the existing summary into the prompt for the model. If there is no existing summary, it defaults to "None". This allows the model to build upon previous context while incorporating new information from the latest messages, ensuring that the long-term memory remains relevant and up-to-date.
    formatted_prompt = summarization_prompt.format(summary=summary if summary else "None")
    
    # Invoke model to generate updated summary
    response = model.invoke([HumanMessage(content=formatted_prompt)] + messages)
    new_summary = response.content
    
    # Optional: Prune old messages from active short-term state to keep context lean,
    # keeping only the last 2 messages for immediate continuity.
    # RemoveMessage is used to indicate which messages should be removed from the active state after summarization. This helps manage memory usage and ensures that the model focuses on the most recent interactions while retaining a semantic summary of the conversation history. The logic here keeps the last two messages for immediate context, while older messages are marked for removal.
    messages_to_remove = [RemoveMessage(id=m.id) for m in messages[:-2]] if len(messages) > 4 else []
    
    return {"summary": new_summary, "messages": messages_to_remove}

# 4. Conditional Check to Trigger Summarization
should_summarize = "summarize" if len(sys.argv) > 1 else "continue" # Placeholder logic for trigger

workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_node("summarizer", summarize_conversation)

workflow.add_edge(START, "agent")
# In production, you would route to 'summarizer' based on message count threshold (e.g., > 10 messages)
workflow.add_edge("agent", END) 

graph = workflow.compile()