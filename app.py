import streamlit as st
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Page setup
st.set_page_config(page_title="Free Groq AI Agent", page_icon="⚡")
st.title("⚡ Free AI Agent (Powered by Groq)")

# 1. Fetch Key from Streamlit Secrets or Sidebar Input
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

# 3. Maintain Session Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. User Interaction & Execution
if prompt := st.chat_input("Assign a task to the agent..."):
    # Append human message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Initialize Groq LLM
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0
    )

    # Agent Prompt Template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant capable of using tools to perform tasks."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Construct Agent and Executor
    agent = create_tool_calling_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Display execution status and final output
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        
        # Prepare chat history for agent execution
        formatted_history = []
        for m in st.session_state.messages[:-1]:
            formatted_history.append((m["role"], m["content"]))

        response = agent_executor.invoke(
            {"input": prompt, "chat_history": formatted_history},
            {"callbacks": [st_cb]}
        )
        
        final_answer = response["output"]
        st.markdown(final_answer)

    # Append assistant response
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
