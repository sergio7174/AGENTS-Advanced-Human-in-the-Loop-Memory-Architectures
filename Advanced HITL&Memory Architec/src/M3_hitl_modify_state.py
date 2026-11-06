# Advanced HITL&Memory Architec/src/M3_hitl_modify_state.py
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
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

@tool("delete_system_file", description="Deletes a specified file from the system.")
def delete_system_file(filename: str) -> str:
    return f"Success: File '{filename}' has been permanently deleted."

tools = [delete_system_file]

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)
model_with_tools = model.bind_tools(tools)
checkpointer = MemorySaver()

def call_model(state: MessagesState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)

workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

graph = workflow.compile(checkpointer=checkpointer, interrupt_before=["tools"])

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "hitl_modify_session"}}
    
    query = HumanMessage(content="Please delete the file important_data.json")
    print(f"\nUser > {query.content}")
    
    # Run until the breakpoint before 'tools'
    # The graph.stream() method is used to execute the graph in a streaming fashion, allowing for real-time monitoring of events. The graph will pause before executing the tool node, enabling human intervention for approval. The state of the graph is checked during the interruption to determine the next node to execute and any pending tool calls. The user is prompted to approve or reject the tool execution, and based on their input, the graph either resumes execution or aborts the operation.
    for event in graph.stream({"messages": [query]}, config):
        # pass is used here to indicate that we are not performing any specific action for each event during the streaming execution. The loop will continue until the graph reaches the breakpoint before executing the 'tools' node, at which point it will pause and allow for human intervention.
        pass
        
    # Inspect current state during interruption
    state = graph.get_state(config)
    last_message = state.values["messages"][-1]
    
    if last_message.tool_calls:
        original_args = last_message.tool_calls[0]["args"]
        print(f"\n⏸️ Interrupted! Agent wants to call '{last_message.tool_calls[0]['name']}' with args: {original_args}")
        
        # Simulate human intervention: intercept and change the target filename to a safe backup
        safe_filename = "safe_backup.bak"
        print(f"🔧 Human Intervention: Modifying target filename from '{original_args.get('filename')}' to '{safe_filename}'")
        
        # Modify the tool call arguments in the message object
        last_message.tool_calls[0]["args"]["filename"] = safe_filename
        
        # Update the graph state explicitly with the modified message
        # as_node="agent" tells LangGraph to attribute this state update as if it came from the agent node
        graph.update_state(config, {"messages": [last_message]}, as_node="agent")
        
    print("\n▶️ Resuming graph with modified arguments...")
    for event in graph.stream(None, config):
        pass
        
    final_state = graph.get_state(config)
    print("\n📤 Final Agent Response >")
    print(final_state.values["messages"][-1].content)