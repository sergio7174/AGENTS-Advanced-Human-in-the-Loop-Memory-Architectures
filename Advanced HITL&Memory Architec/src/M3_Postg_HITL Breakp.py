# Advanced HITL&Memory Architec/src/M3_Postg_HITL Breakp.py
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
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import HumanMessage
import psycopg

load_dotenv()

# 1. Database Connection Configuration
DB_URI = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/agent_db")

# 2. Define a Critical Tool
@tool("execute_database_migration", description="Applies a destructive schema migration to the production database.")
def execute_database_migration(sql_query: str) -> str:
    """Simulates a critical database action."""
    return f"Success: Executed migration query -> {sql_query}"

tools = [execute_database_migration]

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)
model_with_tools = model.bind_tools(tools)

# 3. Setup PostgresSaver Checkpointer
# Using context manager to handle database connection lifecycle safely
with psycopg.connect(DB_URI, autocommit=True) as conn:
    checkpointer = PostgresSaver(conn)
    # Automatically creates required checkpoint tables if they don't exist yet
    checkpointer.setup()

    # 4. Build Graph Workflow
    def call_model(state: MessagesState):
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    # Compile graph with PostgreSQL checkpointer and HITL interrupt
    graph = workflow.compile(checkpointer=checkpointer, interrupt_before=["tools"])

    if __name__ == "__main__":
        print("\n==================================================")
        print("🐘 POSTGRESQL + HITL PERSISTENCE TEST")
        print("==================================================")
        
        config = {"configurable": {"thread_id": "postgres_hitl_thread_99"}}
        
        query = HumanMessage(content="Drop the legacy users table using execute_database_migration.")
        print(f"\nUser > {query.content}")
        
        # Run graph until it hits the breakpoint before 'tools'
        print("\n[Running graph - state will be checkpointed in Postgres...]")
        for event in graph.stream({"messages": [query]}, config):
            pass
            
        # Inspect state pulled directly from PostgreSQL
        state = graph.get_state(config)
        print(f"\n⏸️ Graph Paused! Current next node: {state.next}")
        
        last_message = state.values["messages"][-1]
        if last_message.tool_calls:
            print(f"🔍 Pending Tool: {last_message.tool_calls[0]['name']}")
            print(f"   Args: {last_message.tool_calls[0]['args']}")
            
        approval = input("\n👉 Do you approve this database migration? (yes/no): ").strip().lower()
        
        if approval == "yes":
            print("\n✅ Approved. Resuming execution from PostgreSQL checkpoint...")
            for event in graph.stream(None, config):
                pass
                
            final_state = graph.get_state(config)
            print("\n📤 Final Agent Response >")
            print(final_state.values["messages"][-1].content)
        else:
            print("\n❌ Rejected. Execution halted safely.")