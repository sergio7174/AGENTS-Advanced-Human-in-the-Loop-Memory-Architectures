# Advanced HITL&Memory Architec/src/M3_isolated_memory.py

import os
import psycopg
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/agent_db")

# 1. Multi-Tenant Isolated Memory Table is use to ensure that each user's semantic memory is strictly scoped to their own data, preventing any cross-user data leakage. The table is partitioned by user_id and namespace, allowing for multiple users to interact with the same application without sharing memory.
def init_isolated_memory_table():
    """Create a multi-tenant semantic memory table partitioned by user_id."""
    with psycopg.connect(DB_URI, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tenant_semantic_memory (
                    user_id TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    semantic_summary TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, namespace)
                );
            """)
# 2. Strictly Scoped Memory Operations are provided to ensure that each user's memory is isolated and cannot be accessed by other users. The save_user_memory function upserts the semantic summary for a specific user and namespace, while the load_user_memory function retrieves the memory for a specific user and namespace, ensuring that no cross-user data leakage occurs.
def save_user_memory(user_id: str, namespace: str, summary: str):
    """Upsert memory strictly scoped to a specific user and application namespace."""
    with psycopg.connect(DB_URI, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO tenant_semantic_memory (user_id, namespace, semantic_summary, updated_at)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, namespace) 
                DO UPDATE SET semantic_summary = EXCLUDED.semantic_summary, 
                              updated_at = CURRENT_TIMESTAMP;
            """, (user_id, namespace, summary))
    print(f"🔒 [Isolated DB] Saved memory for User: {user_id} [Namespace: {namespace}]")
# 3. Strict Querying for User Memory is implemented to ensure that when a user retrieves their memory, it is strictly filtered by their user_id and the application namespace. This prevents any possibility of one user accessing another user's memory, maintaining strict data isolation and privacy.
def load_user_memory(user_id: str, namespace: str) -> str:
    """Strictly query memory filtered by user_id to prevent data leaks."""
    with psycopg.connect(DB_URI) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT semantic_summary FROM tenant_semantic_memory WHERE user_id = %s AND namespace = %s;",
                (user_id, namespace)
            )
            row = cur.fetchone()
            return row[0] if row else ""

if __name__ == "__main__":
    init_isolated_memory_table()
    
    # Simulate two different users interacting with the same app namespace ("gym_assistant")
    save_user_memory("user_alice_99", "gym_assistant", "Alice prefers morning yoga and has a knee injury.")
    save_user_memory("user_bob_88", "gym_assistant", "Bob is training for a marathon and tracks heavy lifting.")
    
    # Alice loads her memory — guaranteed isolation
    alice_memory = load_user_memory("user_alice_99", "gym_assistant")
    print(f"\nAlice's Loaded Memory: {alice_memory}")