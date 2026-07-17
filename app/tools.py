from sqlalchemy import create_engine, text, Engine
import pandas as pd
from app.config import settings
from langchain_core.tools import tool



class ServerSession:
    """A session for server-side state management and operations. 
    
    In practice, this would be a separate service from where the agent is running and the agent would communicate with it using a REST API. In this simplified example, we use it to persist the db engine and data returned from the query_db tool.
    """
    def __init__(self):
        self.engine: Engine = None
        self.df: pd.DataFrame = None

        self.engine = self._get_engine()

    def _get_engine(self):
        if self.engine is None:
            # Configure SQLAlchemy for session pooling
            _engine = create_engine(
                settings.SUPABASE_URL,
                pool_size=5,                # Smaller pool size since the pooler manages connections
                max_overflow=5,             # Fewer overflow connections needed
                pool_timeout=10,            # Shorter timeout for getting connections
                pool_recycle=1800,          # Recycle connections more frequently
                pool_pre_ping=True,         # Keep this to verify connections
                pool_use_lifo=True,         # Keep LIFO to reduce number of open connections
                connect_args={
                    "application_name": "onlyvans_agent",
                    "options": "-c statement_timeout=30000",
                    # Keepalives less important with transaction pooler but still good practice
                    "keepalives": 1,
                    "keepalives_idle": 60,
                    "keepalives_interval": 30,
                    "keepalives_count": 3
                }
            )
            return _engine
        return self.engine


# Create a global instance of the ServerSession
session = ServerSession()
print("ServerSession initialized with engine:", session.engine)

@tool
def query_db(query: str) -> pd.DataFrame:
    """Query the database and return the results as a DataFrame."""
    if session.engine is None:
        raise ValueError("Database engine is not initialized.")
    
    with session.engine.connect() as connection:
        result = connection.execute(text(query))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        session.df = df  # Store the DataFrame in the session
        return df

@tool
def generate_visualization(df: pd.DataFrame) -> str:
    """Generate a visualization from the DataFrame and return a description."""
    if df is None or df.empty:
        return "No data available to visualize."
    
    # For simplicity, we just return a string description of the DataFrame
    # In practice, you would generate an actual visualization (e.g., using matplotlib or seaborn)
    return f"Generated a visualization for the DataFrame with {len(df)} rows and {len(df.columns)} columns."


