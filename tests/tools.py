from uuid import uuid4
from app.tools import generate_visualization

result = generate_visualization.func(
    name="transactions_over_time",
    sql_query="""
    SELECT transaction_date
    FROM scout.transactions;
    """,
    plotly_code="""
df["transaction_date"] = pd.to_datetime(df["transaction_date"])

df_grouped = (
    df.groupby(df["transaction_date"].dt.to_period("M"))
      .size()
      .reset_index(name="count")
)

df_grouped["transaction_date"] = (
    df_grouped["transaction_date"].dt.to_timestamp()
)

fig = px.line(
    df_grouped,
    x="transaction_date",
    y="count"
)
""",
    tool_call_id="test-id",
)

print(result)