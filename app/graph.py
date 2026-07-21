from typing import Annotated, NotRequired, Required, TypedDict, Generator
from langchain_core.messages import AIMessageChunk, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
# from langgraph.checkpoint.memory import MemorySaver
from app.tools import query_db, generate_visualization
from langgraph.graph.state import CompiledStateGraph
from app.config import settings

class ScoutState(TypedDict, total=False):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    chart_json: NotRequired[str | None]

class ScoutAgent:
    """
    Represents the Scout agent.
    """
    def __init__(
            self,
            name: str,
            model_name: str = "gemini-3.1-flash-lite",
            tools: list | None = None,
            system_prompt: str = "You are Scout, a data visualization agent. You can query a database and generate visualizations based on the data.",
            temperature: float = 0.1
            ):
        self.name = name
        self.model_name = model_name
        self.system_prompt = SystemMessage(content=system_prompt)
        self.tools = tools or [query_db, generate_visualization]

        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=settings.GOOGLE_API_KEY,
            temperature=temperature
            ).bind_tools(self.tools)
        
        self.runnable = self.build_graph()

    def build_graph(self) -> CompiledStateGraph:
        """
        Build the graph for the Scout agent.
        """

        def scout_node(state: ScoutState):
            response = self.llm.invoke(
                [self.system_prompt, *state["messages"]]
            )

            return {
                "messages": [response]
            }
    
        def router(state: ScoutState):
            """
            Decide whether to execute tools or finish the graph.
            """

            last_message = state["messages"][-1]
            tool_calls = getattr(last_message, "tool_calls", None)

            # Did the LLM request any tool calls?
            if tool_calls:
                return "tools"

            return END
        
        builder = StateGraph(ScoutState)

        builder.add_node("scout", scout_node)
        builder.add_node("tools", ToolNode(self.tools))

        builder.add_edge(START, "scout")
        builder.add_conditional_edges("scout", router, ["tools", END])
        builder.add_edge("tools", "scout")

        return builder.compile()


    def inspect_graph(self):
        """
        Visualize the graph using the mermaid.ink API.
        """
        from IPython.display import display, Image

        graph = self.runnable
        display(Image(graph.get_graph(xray=True).draw_mermaid_png()))

    def invoke(self, message: str, **kwargs) -> str:
        """
        Invoke the Scout agent.
        """
        result = self.runnable.invoke(
            {
                "messages": [HumanMessage(content=message)]
            },
            **kwargs
        )

        return result["messages"][-1].content if result["messages"] else ""

    def stream(self, message: str, **kwargs) -> Generator[str, None, None]:
        """
        stream the output of the Scout agent.
        """
        for chunk, _metadata in self.runnable.stream(
            {"messages": [HumanMessage(content=message)]},
            stream_mode="messages",
            **kwargs,
        ):
            
            if not isinstance(chunk, AIMessageChunk):
                continue

            tool_chunks = getattr(chunk, "tool_call_chunks", None)

            if tool_chunks:
                tool = tool_chunks[0]
                tool_name = tool.get('name', None)
                tool_args = tool.get('args', None)

                # Print tool call
                if tool_name:
                    yield f"\n\n< TOOL CALL: {tool_name} >\n\n"
        
                if tool_args:
                    yield tool_args

                continue

            content = chunk.content

            if isinstance(content, str):
                yield content
            else:
                yield "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict)
                )

system_prompt = """
You are Scout, an AI Business Analyst.

Your purpose is to help users understand and analyze business data stored in relational databases.

You are NOT a chatbot that guesses answers.

You are an investigator.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRIMARY RESPONSIBILITIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your responsibilities are:

• Understand the user's business question.
• Understand the database schema.
• Identify which tables and columns are relevant.
• Create a logical investigation plan.
• Use available tools to retrieve information.
• Base every conclusion on actual database results.
• Explain findings in clear business language.

Never invent information.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKING PRINCIPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Always think before acting.

Do NOT immediately generate SQL.

Instead, follow this workflow.

STEP 1
Understand the user's intent.

Ask yourself:

- What does the user actually want?
- What business metric is involved?
- Is anything ambiguous?

STEP 2
Understand the database.
Always fully qualify table names with their schema.

Use the available schema tool to understand:

- tables
- columns
- relationships
- primary keys
- foreign keys

Never assume the schema.

STEP 3
Create an investigation plan.

Think through:

- Which tables are needed?
- Which joins are required?
- Which filters are required?
- Which aggregations are needed?
- Which business entities are involved?

STEP 4
Execute the investigation.

Use the available tools.

Never fabricate results.

STEP 5
Analyze the returned data.

Do not simply repeat numbers.

Explain:

- what happened
- why it happened (only if supported)
- important patterns
- anomalies
- business meaning

STEP 6
Present the answer clearly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Never hallucinate.

Never invent:

- tables
- columns
- relationships
- business metrics
- values
- SQL results

If information is missing,
say so.

If the question is ambiguous,
ask for clarification.

If multiple interpretations exist,
state them.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATABASE UNDERSTANDING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before writing SQL you should understand the schema.

You should identify:

• relevant tables

• relevant columns

• relationships

• business entities

• join paths

Only after understanding the schema should SQL be generated.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE STYLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Be concise.

Use simple business language.

Avoid unnecessary technical jargon.

Support conclusions using evidence from the database.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR IDENTITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You behave like an experienced Business Analyst.

You investigate before answering.

You verify before concluding.

You explain before recommending.

Accuracy is always more important than speed.
"""

agent = ScoutAgent(
    name="Scout",
    model_name="gemini-3.1-flash-lite",
    system_prompt=system_prompt
).runnable