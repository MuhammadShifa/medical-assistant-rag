import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

from src import config


def ensure_db_exists():
    """Ensure PostgreSQL DB exists; create if missing."""
    conn = psycopg2.connect(
        host=config.PG_HOST,
        port=config.PG_PORT,
        user=config.PG_USER,
        password=config.PG_PASSWORD,
        dbname="postgres",
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (config.PG_DB,))
        if not cur.fetchone():
            cur.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(config.PG_DB))
            )
    conn.close()


def get_conn():
    ensure_db_exists()
    return psycopg2.connect(
        host=config.PG_HOST,
        port=config.PG_PORT,
        user=config.PG_USER,
        password=config.PG_PASSWORD,
        dbname=config.PG_DB,
    )


def init_chat_table():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                role VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                friendly_name VARCHAR(255),
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
            )
        conn.commit()


def save_message(session_id, role, content, friendly_name=None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
            INSERT INTO chat_messages (session_id, role, content, friendly_name)
            VALUES (%s, %s, %s, %s)
            """,
                (session_id, role, content, friendly_name),
            )
        conn.commit()


def load_messages(session_id):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
            SELECT role, content, created_at
            FROM chat_messages
            WHERE session_id = %s
            ORDER BY created_at ASC
            """,
                (session_id,),
            )
            return cur.fetchall()


def list_sessions(limit=10):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT session_id, 
                       COALESCE(friendly_name, session_id) as friendly_name, 
                       MAX(created_at) as last_update
                FROM chat_messages
                GROUP BY session_id, friendly_name
                ORDER BY last_update DESC
                LIMIT %s
            """,
                (limit,),
            )
            return cur.fetchall()


def delete_session(session_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chat_messages WHERE session_id = %s", (session_id,)
            )
        conn.commit()
