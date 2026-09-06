import streamlit as st
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.callbacks.streamlit import StreamlitCallbackHandler

# Page Configuration
st.set_page_config(page_title="AI Agent with Tool Calling", page_icon="🤖")
st.title("🤖 AI Agent with Tools")

# 1. API Key Input via Sidebar
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    if not api_key:
        st.info("Please enter your OpenAI API key to continue.")

# 2. Define Tools
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

# 3. Maintain Chat History State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Handle User Input & Execution
if prompt := st.chat_input("Ask a question or assign a task..."):
    if not api_key:
        st.error("Missing OpenAI API Key.")
        st.stop()

    # Append user input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Initialize model and agent
    llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0)
    agent = create_agent(model=llm, tools=tools)

    # Generate response with tool execution status container
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        
        # Convert session messages to tuple format expected by LangChain agent
        history = [(m["role"], m["content"]) for m in st.session_state.messages]
        
        response = agent.invoke(
            {"messages": history},
            config={"callbacks": [st_cb]}
        )
        
        final_answer = response["messages"][-1].content
        st.markdown(final_answer)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": final_answer})