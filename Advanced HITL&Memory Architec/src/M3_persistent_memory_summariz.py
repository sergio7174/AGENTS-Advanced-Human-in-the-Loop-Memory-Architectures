# Advanced HITL&Memory Architec/src/M3_persistent_memory_summariz.py

import os
import psycopg
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/agent_db")

def init_memory_table():
    """Ensure a dedicated long-term memory table exists in PostgreSQL."""
    with psycopg.connect(DB_URI, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_semantic_memory (
                    user_id TEXT PRIMARY KEY,
                    semantic_summary TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

def save_long_term_memory(user_id: str, summary: str):
    """Upsert the reflection summary into PostgreSQL for cross-session access."""
    with psycopg.connect(DB_URI, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_semantic_memory (user_id, semantic_summary, updated_at)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id) 
                DO UPDATE SET semantic_summary = EXCLUDED.semantic_summary, 
                              updated_at = CURRENT_TIMESTAMP;
            """, (user_id, summary))
    print(f"\n💾 [Postgres Memory] Successfully persisted long-term summary for user: {user_id}")

def load_long_term_memory(user_id: str) -> str:
    """Retrieve long-term memory when a user starts any new chat thread."""
    with psycopg.connect(DB_URI) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT semantic_summary FROM user_semantic_memory WHERE user_id = %s;", (user_id,))
            row = cur.fetchone()
            return row[0] if row else ""

if __name__ == "__main__":
    init_memory_table()
    
    # Simulate an agent finishing a reflection loop and saving the summary
    user = "sergio_moncada"
    updated_summary = "User is a systems engineer building LangGraph agents with Postgres checkpointers and HITL safety."
    
    save_long_term_memory(user, updated_summary)
    
    # Simulate starting a brand-new thread next week
    retrieved_memory = load_long_term_memory(user)
    print(f"\n🧠 [New Thread Loaded] Retrieved memory: {retrieved_memory}")