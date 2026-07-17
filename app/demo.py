from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from app.config import settings

# Tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# LLM with bound tool
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", api_key=settings.GOOGLE_API_KEY, temperature=0.1)
llm_with_tools = llm.bind_tools([multiply])

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", END)

# Compile graph
graph = builder.compile()

def invoke_graph(message: str, **kwargs):
    """
    Invoke the graph with a message and stream the output.
    """

    for chunk, _ in graph.stream(
        {"messages": [HumanMessage(content=message)]},
        stream_mode="messages",
        **kwargs,
    ):
        if not isinstance(chunk, AIMessageChunk):
            continue

        tool_chunks = getattr(chunk, "tool_call_chunks", [])

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
        

for x in invoke_graph("Multiply 6 and 7"):
    print(x, end="")

