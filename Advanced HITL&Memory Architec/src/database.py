# my_langgraph_agent01/src/database.py
import os
import psycopg
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv(".env")

DB_URI = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/agent_db")

# Create a shared connection pool for Postgres
pool = ConnectionPool(
    conninfo=DB_URI,
    min_size=1,
    max_size=10,
    # kwargs are passed to psycopg.connect() for connection configuration, including autocommit mode and prepare statement threshold.
    kwargs={"autocommit": True, "prepare_threshold": 0}
)

def get_checkpointer():
    """Returns the Postgres checkpointer instance backed by the connection pool."""
    return PostgresSaver(pool)

def get_vector_store():
    """Returns the PostgresStore instance backed by the connection pool using Gemini embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        output_dimensionality=768  # Fits safely within PostgreSQL's 2000-dimension HNSW limit
    )
    return PostgresStore(
        pool,
        index={
            "dims": 768,
            "embed": embeddings,
            "fields": ["text"],
        }
    )

def setup_database():
    """Ensures database exists, enables pgvector, and sets up tables."""
    base_uri = DB_URI.rsplit("/", 1)[0] + "/postgres"
    print("Connecting to Postgres server to check/create 'agent_db'...")
    
    # Check if the database exists; if not, create it
    # The code connects to the Postgres server using the base URI (which points to the default 'postgres' database) and checks if the 'agent_db' database exists. If it does not exist, it creates the database. After ensuring the database exists, it connects to 'agent_db' to enable the pgvector extension, which is necessary for vector storage and retrieval. Finally, it sets up the checkpointer and vector store tables using their respective setup methods.
    with psycopg.connect(base_uri, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = 'agent_db';")
            exists = cur.fetchone()
            if not exists:
                print("Creating database 'agent_db'...")
                cur.execute("CREATE DATABASE agent_db;")
                print("Database 'agent_db' created successfully.")
            else:
                print("Database 'agent_db' already exists.")

    print("Connecting to 'agent_db' to enable pgvector extension...")
    with psycopg.connect(DB_URI, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("Vector extension enabled successfully.")

    print("Setting up checkpointer and vector store tables...")
    # checkpointer is an instance of PostgresSaver that manages the saving of short-term conversation thread history to the Postgres database. It ensures that the state of the conversation can be preserved across sessions and application restarts. The setup method creates the necessary tables in the database for storing this information.
    checkpointer = get_checkpointer()
    # checkpointer.setup() initializes the checkpointer by creating the required tables in the Postgres database to store short-term conversation history. This allows the agent to maintain context across sessions and recover from interruptions.
    checkpointer.setup()
    # store is an instance of PostgresStore that manages the storage and retrieval of long-term memory associated with user interactions. It uses the pgvector extension to store vector embeddings of user messages and agent responses, enabling semantic search and context-aware responses. The setup method creates the necessary tables in the database for storing this information.
    store = get_vector_store()
    # store.setup() initializes the vector store by creating the required tables in the Postgres database to store long-term memory. This allows the agent to save and retrieve user memories across sessions, enabling personalized and context-aware interactions.
    store.setup()

if __name__ == "__main__":
    print("Starting database setup...")
    setup_database()
    print("Database setup complete!")