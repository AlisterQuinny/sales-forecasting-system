"""
utils/db.py — SQLAlchemy database connection helper.
"""
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()


def get_engine():
    """Return a SQLAlchemy engine from environment variables."""
    host     = os.getenv("DB_HOST", "localhost")
    port     = os.getenv("DB_PORT", "3306")
    name     = os.getenv("DB_NAME", "sales_db")
    user     = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "")
    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}?charset=utf8mb4"
    return create_engine(url, pool_pre_ping=True)


def run_query(sql: str, params: dict | None = None):
    """Execute a raw SQL query and return rows as a list of dicts."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        return [dict(row._mapping) for row in result]
