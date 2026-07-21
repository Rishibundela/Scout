from typing import Annotated, NotRequired, Required, TypedDict, Generator
from langchain_core.messages import AIMessageChunk, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
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

        return builder.compile(checkpointer=MemorySaver())


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

        
if __name__ == "__main__":
    agent = ScoutAgent(
        name="Scout",
        model_name="gemini-3.1-flash-lite",
    )

    print(agent.invoke(
        message = "Write something long enough so i can check the streaming output of the agent. I want to see if it can handle long messages and stream them properly.",
        config = {"configurable": {"thread_id": "1"}}
        )
    )