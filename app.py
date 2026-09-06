import streamlit as st
import streamlit.components.v1 as components
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq

# 1. Page Configuration & Custom CSS Styling
st.set_page_config(page_title="Zeeshan Ai Agent", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    
    /* Header Styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #9333EA, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Custom Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }

    /* Style Action Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #DC2626, #EF4444);
        color: white;
        border: none;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #B91C1C, #DC2626);
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# 2. Sidebar Setup & History Management
with st.sidebar:
    st.image("https://groq.com/wp-content/uploads/2024/03/PBG-mark-orange.svg", width=50)
    st.title("Control Panel")
    
    # API Key Retrieval
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        api_key = st.text_input("Groq API Key", type="password")
        if not api_key:
            st.warning("Enter API key to activate agent.")
            st.stop()

    st.markdown("---")
    
    # Clear Chat History Functionality
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# 3. Custom Tools
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

# 4. Chat State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title Banner
st.markdown('<div class="main-title">⚡ Ai Agent Tool</div>', unsafe_allow_html=True)
st.caption("Powered by Groq & LangChain • Created By Zeeshan Thaheem")
st.markdown("---")

# Render Past Messages
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg.content)

# 5. User Input & Processing
if prompt := st.chat_input("Assign a task to your agent..."):
    # Append & render human prompt
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Model Initialization
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0
    )
    model_with_tools = llm.bind_tools(tools)

    # Process Agent Response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Processing request..."):
            response = model_with_tools.invoke(st.session_state.messages)
            st.session_state.messages.append(response)

            # Check for tool execution request
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]

                    with st.status(f"🛠️ Tool Called: `{tool_name}`", expanded=True) as status:
                        st.write("**Parameters:**", tool_args)
                        
                        # Execute Tool
                        selected_tool = tools_dict[tool_name]
                        tool_output = selected_tool.invoke(tool_args)
                        
                        st.write("**Output:**", tool_output)
                        status.update(label=f"✅ Tool `{tool_name}` finished", state="complete", expanded=False)
                    
                    # Store tool response
                    tool_msg = ToolMessage(content=str(tool_output), tool_call_id=tool_id)
                    st.session_state.messages.append(tool_msg)

                # Final LLM Synthesis
                final_response = model_with_tools.invoke(st.session_state.messages)
                st.session_state.messages.append(final_response)
                st.markdown(final_response.content)
            else:
                st.markdown(response.content)

    # 6. JavaScript Auto-Scroll to Bottom on New Messages
    components.html(
        """
        <script>
            window.parent.document.querySelector('section.main').scrollTo({
                top: window.parent.document.querySelector('section.main').scrollHeight,
                behavior: 'smooth'
            });
        </script>
        """,
        height=0,
    )
