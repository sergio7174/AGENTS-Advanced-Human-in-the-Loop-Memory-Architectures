# Advanced HITL&Memory Architec/src/M3_Add_Approv_Breakp_HITL.py
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
from langchain_core.messages import HumanMessage
from utils.graph_utils import save_graph_image

load_dotenv()

# 1. Define a "Critical" Tool
@tool("delete_system_file", description="Deletes a specified file from the production system. Use with extreme caution.")
def delete_system_file(filename: str) -> str:
    """Simulates a destructive operation."""
    return f"Success: File '{filename}' has been permanently deleted."

tools = [delete_system_file]

# 2. Initialize Model & Checkpointer
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)
model_with_tools = model.bind_tools(tools)

# Checkpointer is MANDATORY for HITL so state is preserved during pause
checkpointer = MemorySaver()

# 3. Build Graph Workflow
def call_model(state: MessagesState):
    # The call_model function is responsible for invoking the language model with the current state of messages. It takes a MessagesState object as input, which contains the messages exchanged between the user and the model. The function calls the model_with_tools.invoke() method, passing in the list of messages from the state. The model processes these messages and generates a response, which is then returned as a dictionary containing the updated list of messages. This function serves as the core interaction point between the user and the model within the workflow, enabling dynamic responses based on user input and model output.
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Create a StateGraph with MessagesState to manage the flow of messages between the user and the model. The StateGraph is initialized with MessagesState, which allows it to maintain the state of messages throughout the workflow. This setup is essential for handling interactions in a structured manner, enabling the system to process incoming messages, generate responses, and manage the execution flow based on user input and model output.
workflow = StateGraph(MessagesState)
# Add a node to the workflow that calls the model. This node will handle the interaction with the language model, processing incoming messages and generating responses. The call_model function is defined to take the current state of messages, invoke the model with those messages, and return the model's response wrapped in a dictionary format. This node serves as the central point for communication between the user and the model within the workflow.
workflow.add_node("agent", call_model)
# Add a ToolNode to the workflow that contains the critical tool defined earlier. This node will be executed conditionally based on the agent's response, allowing for human intervention before any destructive action is taken.
# What is a ToolNode? A ToolNode is a specialized node in a workflow that encapsulates one or more tools (functions) that can be executed as part of the workflow. In this context, the ToolNode is used to represent the critical tool (delete_system_file) that performs a potentially destructive operation. By using a ToolNode, the workflow can manage the execution of the tool in a controlled manner, allowing for human oversight and approval before the tool is executed. This is particularly important in Human-in-the-Loop (HITL) workflows, where human intervention is necessary to prevent unintended consequences from automated actions.
workflow.add_node("tools", ToolNode(tools))
# Add edges to the workflow to define the execution flow. The first edge connects the START node to the "agent" node, indicating that the workflow begins with the agent's response. The second edge is a conditional edge that connects the "agent" node to either the "tools" node or the END node based on the tools_condition. If the condition is met (i.e., if the agent's response indicates that a tool should be executed), the workflow proceeds to the "tools" node. Otherwise, it goes directly to the END node. The final edge connects the "tools" node back to the "agent" node, allowing for further interaction after tool execution.
workflow.add_edge(START, "agent")
# Add a conditional edge from the "agent" node to either the "tools" node or the END node based on the tools_condition. This allows the workflow to branch based on the agent's response, enabling human intervention before executing any critical tools. If the condition is met, the workflow proceeds to the "tools" node for potential execution of the critical tool. If not, it goes directly to the END node, concluding the workflow without executing any destructive actions.
workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
# Add an edge from the "tools" node back to the "agent" node, allowing for further interaction after tool execution. This creates a loop in the workflow where, after executing a tool (if approved), the workflow can return to the agent for additional processing or responses. This design supports iterative interactions and ensures that the workflow can continue seamlessly after any tool execution, maintaining the flow of communication between the user and the model.
workflow.add_edge("tools", "agent")

# 4. Compile with interrupt_before=["tools"]
# This guarantees execution freezes right before the ToolNode runs!
# The graph is compiled with the interrupt_before parameter set to ["tools"], which ensures that the execution of the graph will pause right before the ToolNode is executed. This allows for human intervention and approval before any potentially destructive operations are carried out by the tool. By pausing at this point, the system can prompt the user for confirmation, ensuring that critical actions are not performed without explicit consent. This is a key feature of Human-in-the-Loop (HITL) workflows, where human oversight is necessary to prevent unintended consequences.
graph = workflow.compile(checkpointer=checkpointer, interrupt_before=["tools"])

if __name__ == "__main__":

    # Call the external utility function, passing the compiled graph
    # Test invoking the graph directly
    print("\n--- Generating Graph Image ---")
    save_graph_image(graph, output_filename="Add_Approv_Breakp_HITL.png")

    print("\n========================================")
    print("🛡️ HITL APPROVAL BREAKPOINT TEST")
    print("========================================")
    
    config = {"configurable": {"thread_id": "hitl_session_01"}}
    
    # User triggers a critical request
    query = HumanMessage(content="Please delete the file production_config.json immediately.")
    # The graph will pause before executing the tool, allowing for human approval.
    # query.content = "Please delete the file production_config.json immediately."
    print(f"\nUser > {query.content}")
    
    # Run the graph (it will automatically pause before executing 'delete_system_file')
    print("\n[Running graph...]")
    # The graph.stream() method is used to execute the graph in a streaming fashion, allowing for real-time monitoring of events. The graph will pause before executing the tool node, enabling human intervention for approval. The state of the graph is checked during the interruption to determine the next node to execute and any pending tool calls. The user is prompted to approve or reject the tool execution, and based on their input, the graph either resumes execution or aborts the operation.
    # what kind of events in graph could I find? graph.stream(), what others will i use to monitor the graph execution? graph.stream() is the primary method for monitoring the execution of a StateGraph in real-time. It allows you to receive events as they occur during the graph's execution, which can include:
    # - Node Execution Events: Notifications when a node starts and finishes executing.
    # graph.invike() is another method that can be used to execute the graph, but it does not provide real-time event monitoring like graph.stream(). It is typically used for synchronous execution where you want to wait for the entire graph to complete before proceeding. In contrast, graph.stream() allows for more interactive and responsive handling of the graph's execution, especially useful in scenarios like Human-in-the-Loop (HITL) where user intervention may be required.
    for event in graph.stream({"messages": [query]}, config):
        print(f"Event: {list(event.keys())}")
        
    # Check graph state during the interruption
    # The graph.get_state(config) method retrieves the current state of the graph execution, allowing you to inspect the next node to be executed and any relevant values, such as messages or tool calls. This is particularly useful during interruptions, as it provides insight into what the graph is waiting for and what actions are pending. In this case, it helps determine if there are any tool calls that require human approval before proceeding with execution.
    state = graph.get_state(config)
    print(f"\n⏸️ Graph Paused! Next node to execute: {state.next}")
    # The state.values dictionary contains the current values of the graph's state, including any messages that have been processed or generated. By checking if there are any messages in state.values["messages"], you can determine if there are pending tool calls that require human approval. If a tool call is detected, the details of the tool call (such as its name and arguments) are printed to inform the user of the action that is about to be executed.
    if state.values["messages"]:
        # The last_message variable retrieves the most recent message from the state.values["messages"] list. If this message contains any tool calls (indicating that the agent is requesting to execute a tool), the details of the first tool call are printed, including its name and arguments. This provides context to the human reviewer about what action is being requested and allows them to make an informed decision about whether to approve or reject the execution of the tool.
        last_message = state.values["messages"][-1]
        # If the last message contains any tool calls, it indicates that the agent is requesting to execute a tool. The code checks if there are any tool calls in last_message.tool_calls and, if so, prints the name of the first tool call and its arguments. This information is presented to the human reviewer to help them understand what action is being requested and to facilitate their decision-making process regarding approval or rejection of the tool execution.
        if last_message.tool_calls:
            # The print statements provide a clear indication to the human reviewer that a pending tool call has been detected, along with the specific tool name and its associated arguments. This allows the reviewer to assess the potential impact of executing the tool and make an informed decision about whether to approve or reject the action. By presenting this information in a concise manner, the code ensures that the human reviewer has all the necessary context to make a responsible choice regarding the execution of potentially destructive operations.
            print(f"🔍 Pending Tool Call Detected: {last_message.tool_calls[0]['name']}")
            # The print statement displays the arguments associated with the pending tool call, providing additional context to the human reviewer. By showing the specific parameters that will be used in the tool execution, the reviewer can better understand the potential consequences of approving or rejecting the action. This transparency is crucial for responsible decision-making, especially when dealing with critical operations that could have significant impacts on the system or data.
            print(f"   Arguments: {last_message.tool_calls[0]['args']}")
            
    # Simulate Human Decision
    approval = input("\n👉 Do you approve this tool execution? (yes/no): ").strip().lower()
    # The input() function prompts the human reviewer to provide their decision regarding the pending tool execution. The input is stripped of leading and trailing whitespace and converted to lowercase to ensure consistent handling of the response. Based on the reviewer's input, the code will either resume the graph execution if approval is granted or abort the operation if approval is denied. This interactive step allows for human oversight and control over potentially destructive actions within the workflow.
    if approval == "yes":
        print("\n✅ Approval granted. Resuming graph execution...")
        # Resume execution by passing None into invoke/stream with the same thread config
        # The graph.stream() method is called again with None as the input, indicating that the graph should continue execution from the point where it was paused. The same configuration (config) is passed to maintain the context of the current thread. This allows the graph to proceed with executing the pending tool call that was approved by the human reviewer, ensuring that the workflow continues seamlessly after the interruption.
        for event in graph.stream(None, config):
            pass
            
        final_state = graph.get_state(config)
        print("\n📤 Final Agent Response >")
        print(final_state.values["messages"][-1].content)
    else:
        print("\n❌ Action rejected by human reviewer. Aborting execution.")