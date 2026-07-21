from sqlalchemy import create_engine, text, Engine
import pandas as pd
from app.config import settings
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from typing import Annotated
from langchain_core.messages import HumanMessage, ToolMessage



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
# print("ServerSession initialized with engine:", session.engine)

@tool
def query_db(query: str) -> str:
    """Query the database using Postgres SQL.

    Args:
        query: The SQL query to execute. Must be a valid postgres SQL string that can be executed directly.

    Returns:
        str: The query result as a markdown table.
    """
    try:
        # Use the global engine in the server session to connect to Supabase
        with session.engine.connect().execution_options(
            isolation_level="READ COMMITTED"
        ) as conn:
            result = conn.execute(text(query))
            columns = list(result.keys())
            rows = result.fetchall()
            df =   pd.DataFrame(rows, columns=columns)
            # store the dataframe in the server session
            session.df = df

        return df.to_markdown(index=False)
    except Exception as e:

        raise RuntimeError(f"Database tool failed: {e}")



# @tool
# def generate_visualization(
#     name: str,
#     sql_query: str,
#     plotly_code: str,
#     tool_call_id: Annotated[str, InjectedToolCallId] 
#     ) -> Command | str:

#     '''
#     Generate a visualization using Python, SQL, and Plotly. If the visualization is successfully generated, it's automatically rendered for the user on the frontend.

#     Args:
#         name: The name of the visualization. Should be a short name with underscores and no spaces.
#         sql_query: The SQL query to retrieve data for the visualization. Must be a valid postgres SQL string that can be executed directly. The query will be executed and the result will be loaded into a DataFrame named 'df'.
#         plotly_code: Python code that generates a Plotly figure. The code should create a variable named 'fig' that contains the Plotly figure object.

#     Returns:
#         str: Success message if successful or an error message.

#     ## Assumptions
#     Assume the data is already loaded into a DataFrame named 'df' and the following libraries are already imported for immediate use: 
    
#     import pandas as pd
#     import plotly.express as px
#     import plotly.graph_objects as go
#     import plotly

#     ## Example:
#     User asks "Show me the top 5 creators by revenue"

#     sql_query = "SELECT c.id, c.first_name, c.last_name, SUM(t.amount_usd) AS total_revenue\nFROM creators c\nJOIN transactions t ON c.id = t.creator_id\nGROUP BY c.id, c.first_name, c.last_name\nORDER BY total_revenue DESC\nLIMIT 5;"
#     plotly_code = "fig = px.bar(df, x='first_name', y='total_revenue', title='Top 5 Creators by Revenue')\nfig.update_layout(xaxis_title='Creator', yaxis_title='Total Revenue ($)')"
#     '''
     
#     import io
#     import os
#     from contextlib import redirect_stdout, redirect_stderr

#     # Create the output direct if it doesn't exist
#     os.makedirs("output", exist_ok=True)
    
#     # set the output file path
#     file_path = f"output/{name}.json"

#     # Capture stdout and stderr
#     stdout_capture = io.StringIO()
#     stderr_capture = io.StringIO()

#     # Add SQL query to the code

#     import textwrap

#     plotly_code = textwrap.dedent(plotly_code).strip()

#     pre_code = textwrap.dedent(f"""
# from sqlalchemy import text
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import plotly.io as pio
# import plotly

# df = pd.read_sql(
#     text(\"\"\"{sql_query}\"\"\"),
#     engine
# )
# """)
    
#     post_code = textwrap.dedent(f"""
# if "fig" in locals() or "fig" in globals():
#     fig_json = pio.to_json(fig)
#     with open("{file_path}", "w") as f:
#         f.write(fig_json)
# """)

#     code = "\n\n".join([
#         pre_code,
#         plotly_code,
#         post_code
#     ])

#     print(code)

#     exec_env = {
#         "engine": session.engine
#     }

#     try:
#         print("=" * 80)
#         print(code)
#         print("=" * 80)

#         with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
#             exec(code, exec_env)

#         # Get the output and error messages
#         print(f"STDOUT: \n\n{stdout_capture.getvalue()}\n")
#         print(f"STDERR: \n\n{stderr_capture.getvalue()}\n")

#         # check if the fig was created and saved to the file
#         if os.path.exists(file_path):
#             with open(file_path, 'r') as f:
#                 fig_json = f.read()
#             return Command(
#                 update={
#                     # Update the state keys
#                     "chart_json": fig_json,
#                     # Update the messages with a ToolMessage indicating success
#                     "messages":[
#                         ToolMessage("Visualization generated successfully.",
#                                     tool_call_id=tool_call_id
#                         )
#                     ]
#                 }
#             )
#         else:
#             raise Exception(f"Error: Failed to generate visualization.\n\n<stderr>\n{stderr_capture.getvalue()}\n</stderr>")
        
#     except Exception as e:
#         # Get the error message
#         error_message = str(e)
#         return f"Error executing visualization code: {error_message}"


@tool
def generate_visualization(
    name: str,
    sql_query: str,
    plotly_code: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command | str:
    """
    Generate a Plotly visualization.

    The LLM should ONLY generate Plotly code.

    Assumptions:
    - DataFrame is named `df`
    - Figure variable must be named `fig`
    """

    import io
    import os
    import textwrap
    from contextlib import redirect_stdout, redirect_stderr

    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    from sqlalchemy import text

    try:

        # -------------------------------
        # Prepare output
        # -------------------------------

        os.makedirs("output", exist_ok=True)

        file_path = f"output/{name}.json"

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # -------------------------------
        # Execute SQL
        # -------------------------------

        df = pd.read_sql(
            text(sql_query),
            session.engine,
        )

        # -------------------------------
        # Clean LLM code
        # -------------------------------

        plotly_code = textwrap.dedent(plotly_code).strip()

        # -------------------------------
        # Execution environment
        # -------------------------------

        exec_env = {
            "df": df,
            "pd": pd,
            "px": px,
            "go": go,
            "pio": pio,
        }

        print("=" * 80)
        print("Generated Plotly Code")
        print("=" * 80)
        print(plotly_code)
        print("=" * 80)

        # -------------------------------
        # Execute Plotly code
        # -------------------------------

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(plotly_code, exec_env)

        # -------------------------------
        # Validate figure
        # -------------------------------

        if "fig" not in exec_env:
            raise RuntimeError(
                "The generated code did not create a variable named 'fig'."
            )

        fig = exec_env["fig"]

        # -------------------------------
        # Save JSON
        # -------------------------------

        fig_json = pio.to_json(fig)

        with open(file_path, "w") as f:
            f.write(fig_json)

        # -------------------------------
        # Success
        # -------------------------------

        return Command(
            update={
                "chart_json": fig_json,
                "messages": [
                    ToolMessage(
                        content="Visualization generated successfully.",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    except Exception as e:

        print("=" * 80)
        print("Visualization Error")
        print("=" * 80)
        print(e)

        if stdout_capture.getvalue():
            print("\nSTDOUT")
            print(stdout_capture.getvalue())

        if stderr_capture.getvalue():
            print("\nSTDERR")
            print(stderr_capture.getvalue())

        return (
            f"Error executing visualization:\n\n"
            f"{type(e).__name__}: {e}\n\n"
            f"STDERR:\n{stderr_capture.getvalue()}"
        )