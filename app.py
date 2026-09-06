import streamlit as st
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq

# Page Setup
st.set_page_config(page_title="Free Groq AI Agent", page_icon="⚡")
st.title("⚡ Tool-Calling AI Agent (Groq)")

# 1. API Key Setup
api_key = st.secrets.get("GROQ_API_KEY")

if not api_key:
    api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
    if not api_key:
        st.info("Get your free API Key at https://console.groq.com")
        st.stop()

# 2. Define Custom Tools
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
tools_dict = {t.name: t for t in tools}

# 3. Maintain Session Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render past messages
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# 4. User Interaction & Tool Loop
if prompt := st.chat_input("Assign a task to the agent..."):
    # Display user query
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Initialize Model & Bind Tools
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0
    )
    model_with_tools = llm.bind_tools(tools)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # First LLM Call
            response = model_with_tools.invoke(st.session_state.messages)
            st.session_state.messages.append(response)

            # Check if the LLM called any tool
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]

                    st.status(f"Executing tool: `{tool_name}` with args: `{tool_args}`", state="running")
                    
                    # Execute tool function locally
                    selected_tool = tools_dict[tool_name]
                    tool_output = selected_tool.invoke(tool_args)
                    
                    # Store tool execution output
                    tool_msg = ToolMessage(content=str(tool_output), tool_call_id=tool_id)
                    st.session_state.messages.append(tool_msg)

                # Final LLM Call to process tool results
                final_response = model_with_tools.invoke(st.session_state.messages)
                st.session_state.messages.append(final_response)
                st.markdown(final_response.content)
            else:
                st.markdown(response.content)
