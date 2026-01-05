"""
Create PostgreSQL tables with pgvector extension
Run this script once to initialize the database schema
"""

from db_config import get_connection_raw


def create_tables():
    # Use raw connection since vector extension doesn't exist yet
    conn = get_connection_raw()
    cur = conn.cursor()

    # Enable pgvector extension FIRST
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()  # Commit extension creation before creating tables

    # Create policy table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS policy (
            id VARCHAR(100) PRIMARY KEY,
            subject TEXT NOT NULL,
            content TEXT NOT NULL
        );
    """
    )

    # Create law_unit table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS law_unit (
            id VARCHAR(100) NOT NULL,
            policy_id VARCHAR(100) NOT NULL,
            content TEXT NOT NULL,
            PRIMARY KEY (id, policy_id),
            FOREIGN KEY (policy_id) REFERENCES policy(id) ON DELETE CASCADE
        );
    """
    )

    # Create hiq table with vector column
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS hiq (
            id SERIAL PRIMARY KEY,
            law_unit_id VARCHAR(100) NOT NULL,
            policy_id VARCHAR(100) NOT NULL,
            hypothetical_question TEXT NOT NULL,
            embedding vector(1024),
            FOREIGN KEY (law_unit_id, policy_id) REFERENCES law_unit(id, policy_id) ON DELETE CASCADE
        );
    """
    )

    # Create law_references table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS law_references (
            law_unit_id VARCHAR(100) NOT NULL,
            policy_id VARCHAR(100) NOT NULL,
            reference VARCHAR(100) NOT NULL,
            PRIMARY KEY (law_unit_id, policy_id, reference),
            FOREIGN KEY (law_unit_id, policy_id) REFERENCES law_unit(id, policy_id) ON DELETE CASCADE
        );
    """
    )

    # Create law_cross_references table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS law_cross_references (
            law_unit_id VARCHAR(100) NOT NULL,
            policy_id VARCHAR(100) NOT NULL,
            cross_reference VARCHAR(100) NOT NULL,
            PRIMARY KEY (law_unit_id, policy_id, cross_reference),
            FOREIGN KEY (law_unit_id, policy_id) REFERENCES law_unit(id, policy_id) ON DELETE CASCADE
        );
    """
    )

    # Create sel_fact table with vector column
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sel_fact (
            id SERIAL PRIMARY KEY,
            policy_id VARCHAR(100) NOT NULL,
            fact TEXT NOT NULL,
            embedding vector(1024),
            FOREIGN KEY (policy_id) REFERENCES policy(id) ON DELETE CASCADE
        );
    """
    )

    # Create indexes for vector similarity search (using cosine distance)
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS hiq_embedding_idx ON hiq 
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS sel_fact_embedding_idx ON sel_fact 
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    """
    )

    conn.commit()
    cur.close()
    conn.close()

    print("Tables created successfully!")


def drop_tables():
    """Drop all tables - use with caution!"""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS law_cross_references CASCADE;")
    cur.execute("DROP TABLE IF EXISTS law_references CASCADE;")
    cur.execute("DROP TABLE IF EXISTS hiq CASCADE;")
    cur.execute("DROP TABLE IF EXISTS sel_fact CASCADE;")
    cur.execute("DROP TABLE IF EXISTS law_unit CASCADE;")
    cur.execute("DROP TABLE IF EXISTS policy CASCADE;")

    conn.commit()
    cur.close()
    conn.close()

    print("Tables dropped successfully!")


if __name__ == "__main__":
    create_tables()
