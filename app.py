
from langchain_core.messages import HumanMessage
from graph_builder import build_graph
import streamlit as st
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=api_key
)

api_key = st.secrets["OPENAI_API_KEY"]

graph = build_graph()

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in {"exit", "quit"}:
        break

    result = graph.invoke({
        "messages": [HumanMessage(content=user_input)]
    })

    print("\nFinal messages:")
    for msg in result["messages"]:
        msg_type = msg.__class__.__name__
        print(f"{msg_type}: {msg.content}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print("  tool_calls:", msg.tool_calls)
