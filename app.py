import streamlit as st
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.callbacks.streamlit import StreamlitCallbackHandler

st.set_page_config(page_title="Free Groq AI Agent", page_icon="⚡")
st.title("⚡ Free AI Agent (Powered by Groq)")

# Get API key from Streamlit Secrets or manual input
api_key = st.secrets.get("GROQ_API_KEY")

if not api_key:
    api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
    if not api_key:
        st.info("Get your free Groq key at https://console.groq.com")
        st.stop()

# Custom Tools
@tool
def calculate_area(length: float, width: float) -> float:
    """Calculates the area of a rectangle given length and width."""
    return length * width

@tool
def check_inventory(item_name: str) -> str:
    """Checks the current stock count of an item in the warehouse."""
    inventory = {"laptop": 15, "phone": 42, "headphones": 0}
    count = inventory.get(item_name.lower(), 0)
    return f"Stock for {item_name}: {count} units available."

tools = [calculate_area, check_inventory]

# Maintain Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Task Processing
if prompt := st.chat_input("Assign a task to the agent..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Initialize Groq LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0
    )
    
    agent = create_agent(model=llm, tools=tools)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        history = [(m["role"], m["content"]) for m in st.session_state.messages]
        
        response = agent.invoke({"messages": history}, config={"callbacks": [st_cb]})
        final_answer = response["messages"][-1].content
        st.markdown(final_answer)

    st.session_state.messages.append({"role": "assistant", "content": final_answer})
