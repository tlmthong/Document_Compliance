"""
Database configuration for PostgreSQL with pgvector
"""

import os
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
import numpy as np

# Database connection settings - can be overridden with environment variables
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "policy_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

DIM = 1024


def get_connection_raw():
    """Create a raw database connection without vector registration (for setup)"""
    conn = psycopg2.connect(**DB_CONFIG)
    return conn


def get_connection():
    """Create a new database connection with vector support"""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        register_vector(conn)
    except psycopg2.ProgrammingError:
        # Vector extension not yet created - this is OK during initial setup
        pass
    return conn


@contextmanager
def get_db_cursor(commit=True):
    """Context manager for database operations"""
    conn = get_connection()
    try:
        cur = conn.cursor()
        yield cur
        if commit:
            conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()


def vector_to_list(vec) -> list:
    """Convert numpy array or pgvector to Python list"""
    if isinstance(vec, np.ndarray):
        return vec.tolist()
    return list(vec)


def list_to_vector(lst) -> np.ndarray:
    """Convert Python list to numpy array for pgvector"""
    return np.array(lst, dtype=np.float32)
