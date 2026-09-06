import json
import streamlit as st
import streamlit.components.v1 as components
from groq import Groq

# 1. Page Configuration & Styling
st.set_page_config(page_title="AI Agent Studio", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #9333EA, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
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

# 2. Sidebar & Key Setup
with st.sidebar:
    st.image("https://groq.com/wp-content/uploads/2024/03/PBG-mark-orange.svg", width=50)
    st.title("Control Panel")
    
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        api_key = st.text_input("Groq API Key", type="password")
        if not api_key:
            st.warning("Enter your Groq API Key to proceed.")
            st.stop()

    st.markdown("---")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

client = Groq(api_key=api_key)

# 3. Tool Definitions
def calculate_area(length: float, width: float) -> str:
    return str(float(length) * float(width))

def check_inventory(item_name: str) -> str:
    inventory = {"laptop": 15, "phone": 42, "headphones": 0}
    count = inventory.get(item_name.lower(), 0)
    return f"Stock for {item_name}: {count} units available."

available_tools = {
    "calculate_area": calculate_area,
    "check_inventory": check_inventory,
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "calculate_area",
            "description": "Calculates the area of a rectangle given length and width.",
            "parameters": {
                "type": "object",
                "properties": {
                    "length": {"type": "number", "description": "The length of the shape"},
                    "width": {"type": "number", "description": "The width of the shape"},
                },
                "required": ["length", "width"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_inventory",
            "description": "Checks the current stock count of an item in the warehouse.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {"type": "string", "description": "Name of the item to look up"},
                },
                "required": ["item_name"],
            },
        },
    },
]

# 4. Initialize Session State & Handle Legacy State Cleaning
if "messages" not in st.session_state:
    st.session_state.messages = []

# Validate structure to prevent TypeError from old session objects
if st.session_state.messages and not isinstance(st.session_state.messages[0], dict):
    st.session_state.messages = []

st.markdown('<div class="main-title">⚡ AI Tool Agent Studio</div>', unsafe_allow_html=True)
st.caption("Powered by Groq Native SDK • Features auto-scroll, history control & local tool execution")
st.markdown("---")

# Render Messages safely
for msg in st.session_state.messages:
    if isinstance(msg, dict):
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        elif role == "assistant" and content:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)

# 5. Chat & Tool Execution
if prompt := st.chat_input("Assign a task to your agent..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Processing request..."):
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=st.session_state.messages,
                tools=tools_schema,
                tool_choice="auto",
            )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            # Store assistant response dictionary safely
            st.session_state.messages.append(response_message.model_dump())

            if tool_calls:
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    with st.status(f"🛠️ Tool Called: `{function_name}`", expanded=True) as status:
                        st.write("**Parameters:**", function_args)
                        
                        function_to_call = available_tools[function_name]
                        function_response = function_to_call(**function_args)
                        
                        st.write("**Output:**", function_response)
                        status.update(label=f"✅ Tool `{function_name}` executed", state="complete", expanded=False)

                    st.session_state.messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    })

                second_response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=st.session_state.messages,
                )
                final_answer = second_response.choices[0].message.content
                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
            else:
                final_answer = response_message.content
                st.markdown(final_answer)

    # Auto-Scroll View
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
