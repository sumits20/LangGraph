from typing import Annotated, TypedDict, Any
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

from tools import get_weather, multiply, get_fun_fact


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    debug_steps: list[dict[str, Any]]


tools = {
    "get_weather": get_weather,
    "multiply": multiply,
    "get_fun_fact": get_fun_fact,
}


def serialize_message(msg: BaseMessage) -> dict:
    data = {
        "type": msg.__class__.__name__,
        "content": msg.content if getattr(msg, "content", None) else ""
    }

    if hasattr(msg, "tool_calls") and msg.tool_calls:
        data["tool_calls"] = msg.tool_calls
    else:
        data["tool_calls"] = []

    return data


def get_llm_with_tools():
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in Streamlit secrets")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key
    )
    return llm.bind_tools(list(tools.values()))


def chatbot_node(state: ChatState) -> dict:
    llm_with_tools = get_llm_with_tools()

    response = llm_with_tools.invoke(state["messages"])

    debug_row = {
        "step_type": "llm",
        "node": "chatbot",
        "input_messages": [serialize_message(m) for m in state["messages"]],
        "output_text": response.content if response.content else "",
        "tool_calls": response.tool_calls if hasattr(response, "tool_calls") else []
    }

    existing_steps = state.get("debug_steps", [])
    return {
        "messages": [response],
        "debug_steps": existing_steps + [debug_row]
    }


def custom_tool_node(state: ChatState) -> dict:
    last_message = state["messages"][-1]
    tool_messages = []
    debug_rows = []

    for call in last_message.tool_calls:
        tool_name = call["name"]
        args = call["args"]
        tool_id = call["id"]

        result = tools[tool_name].invoke(args)

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_id)
        )

        debug_rows.append({
            "step_type": "tool",
            "node": "tools",
            "tool_name": tool_name,
            "tool_args": args,
            "tool_result": str(result)
        })

    existing_steps = state.get("debug_steps", [])
    return {
        "messages": tool_messages,
        "debug_steps": existing_steps + debug_rows
    }


def route_tools(state: ChatState):
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return END


def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("chatbot", chatbot_node)
    graph.add_node("tools", custom_tool_node)

    graph.set_entry_point("chatbot")
    graph.add_conditional_edges("chatbot", route_tools)
    graph.add_edge("tools", "chatbot")

    return graph.compile()
