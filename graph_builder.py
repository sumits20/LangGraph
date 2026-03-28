from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

from tools import get_weather, multiply, get_fun_fact

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


tools = [get_weather, multiply, get_fun_fact]
tool_node = ToolNode(tools)

def get_llm_with_tools():
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in Streamlit secrets")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key
    )
    return llm.bind_tools(tools)

def chatbot_node(state: ChatState) -> dict:
    llm_with_tools = get_llm_with_tools()
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def route_tools(state: ChatState):
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return END


def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("chatbot", chatbot_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("chatbot")
    graph.add_conditional_edges("chatbot", route_tools)
    graph.add_edge("tools", "chatbot")

    return graph.compile()
