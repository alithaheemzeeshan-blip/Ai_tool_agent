import json
import re
import streamlit as st
import streamlit.components.v1 as components
from groq import Groq
from duckduckgo_search import DDGS
import yfinance as yf

# 1. Page Configuration & Custom Styling
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

# 2. Sidebar Setup & Key Management
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

# Dynamic active model retriever
@st.cache_data(ttl=1800)
def get_active_model(key_str):
    try:
        models = client.models.list().data
        active_ids = [m.id for m in models if "whisper" not in m.id and "guard" not in m.id]
        priority = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ]
        for p in priority:
            if p in active_ids:
                return p
        return active_ids[0] if active_ids else "llama-3.3-70b-versatile"
    except Exception:
        return "llama-3.3-70b-versatile"

ACTIVE_MODEL = get_active_model(api_key)

# 3. Tool Implementations
def calculate_area(length: float, width: float) -> str:
    return str(float(length) * float(width))

def check_inventory(item_name: str) -> str:
    inventory = {"laptop": 15, "phone": 42, "headphones": 0}
    count = inventory.get(item_name.lower(), 0)
    return f"Stock for {item_name}: {count} units available."

def web_search(query: str) -> str:
    """Performs web search with yfinance fallback for stock tickers and query sanitization."""
    ticker_match = re.search(r'\b([A-Z]{1,5})\b', query)
    if any(k in query.lower() for k in ["stock", "price", "quote", "market cap"]):
        if ticker_match:
            symbol = ticker_match.group(1).upper()
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.fast_info
                last_price = info.get('lastPrice', None)
                if last_price:
                    return f"Stock Symbol: {symbol}\nCurrent Price: ${last_price:.2f}"
            except Exception:
                pass

    clean_query = query.replace("today", "").replace("stock price", "stock news").strip()
    backends = ["auto", "html", "lite"]
    
    for backend in backends:
        try:
            results = DDGS().text(
                keywords=clean_query, 
                max_results=3, 
                backend=backend
            )
            if results:
                clean_snippets = [
                    f"Title: {r.get('title')}\nSnippet: {r.get('body')}" 
                    for r in results
                ]
                return "\n\n".join(clean_snippets)
        except Exception:
            continue
            
    return f"No web results found for query: '{query}'."

available_tools = {
    "calculate_area": calculate_area,
    "check_inventory": check_inventory,
    "web_search": web_search,
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
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the live internet for recent news, gold prices, outside facts, or real-time information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query terms"},
                },
                "required": ["query"],
            },
        },
    },
]

# 4. State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.session_state.messages and not isinstance(st.session_state.messages[0], dict):
    st.session_state.messages = []

st.markdown('<div class="main-title">⚡ Real-Time AI Agent Studio</div>', unsafe_allow_html=True)
st.caption(f"Powered by Groq Cloud (`{ACTIVE_MODEL}`) • Live Web Search Enabled")
st.markdown("---")

# Render Messages Safely
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

# Strict Message Builder for Groq API payload compliance
def prepare_messages_for_api(messages):
    system_instruction = {
        "role": "system",
        "content": (
            "You are an active tool-using AI agent. "
            "You MUST NEVER state that you cannot access real-time information, check external sites, or lack current data. "
            "Whenever a user asks about real-time news, current events, recent developments, gold prices, weather, or facts, "
            "you MUST call the `web_search` tool."
        )
    }
    cleaned = [system_instruction]
    for m in messages:
        if not isinstance(m, dict):
            continue
        
        role = m.get("role")
        if role not in ["user", "assistant", "tool"]:
            continue

        msg_copy = {"role": role}

        if role == "user":
            msg_copy["content"] = str(m.get("content") or "")

        elif role == "assistant":
            if "tool_calls" in m and m["tool_calls"]:
                msg_copy["tool_calls"] = m["tool_calls"]
                if m.get("content"):
                    msg_copy["content"] = str(m["content"])
            else:
                msg_copy["content"] = str(m.get("content") or "")

        elif role == "tool":
            msg_copy["content"] = str(m.get("content") or "")
            msg_copy["tool_call_id"] = str(m.get("tool_call_id", ""))
            msg_copy["name"] = str(m.get("name", ""))

        cleaned.append(msg_copy)
    return cleaned

# 5. User Interaction & Tool Execution Loop
if prompt := st.chat_input("Ask a real-time question or assign a task..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Processing request..."):
            
            while True:
                api_messages = prepare_messages_for_api(st.session_state.messages)
                
                try:
                    response = client.chat.completions.create(
                        model=ACTIVE_MODEL,
                        messages=api_messages,
                        tools=tools_schema,
                        tool_choice="auto",
                    )
                except Exception:
                    response = client.chat.completions.create(
                        model=ACTIVE_MODEL,
                        messages=api_messages,
                    )

                response_message = response.choices[0].message
                tool_calls = getattr(response_message, "tool_calls", None)

                assistant_dict = {"role": "assistant"}
                if response_message.content:
                    assistant_dict["content"] = response_message.content
                
                if tool_calls:
                    assistant_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in tool_calls
                    ]
                
                st.session_state.messages.append(assistant_dict)

                if not tool_calls:
                    final_answer = response_message.content or "Task completed."
                    st.markdown(final_answer)
                    break

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    with st.status(f"🛠️ Executing Tool: `{function_name}`", expanded=True) as status:
                        st.write("**Parameters:**", function_args)
                        
                        function_to_call = available_tools[function_name]
                        function_response = function_to_call(**function_args)
                        
                        st.write("**Output:**", function_response)
                        status.update(label=f"✅ Tool `{function_name}` finished", state="complete", expanded=False)

                    st.session_state.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": str(function_response),
                    })

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
