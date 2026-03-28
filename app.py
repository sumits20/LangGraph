import streamlit as st
from langchain_core.messages import HumanMessage
from graph_builder import build_graph

st.set_page_config(page_title="Mini Chat BOT - using LangGraph", layout="wide")
st.title("LangGraph Chat BOT")

@st.cache_resource
def get_graph():
    return build_graph()

graph = get_graph()

user_input = st.text_input("Ask something")

if st.button("Run") and user_input:
    result = graph.invoke({
        "messages": [HumanMessage(content=user_input)]
    })

    st.subheader("Final messages")

    for msg in result["messages"]:
        msg_type = msg.__class__.__name__
        st.write(f"**{msg_type}:** {msg.content if msg.content else '(no text content)'}")

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            st.write("Tool calls:")
            st.json(msg.tool_calls)
