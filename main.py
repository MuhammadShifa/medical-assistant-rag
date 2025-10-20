import streamlit as st
import time
import socket
import uuid
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from src import config, utils, retriever, reranker
from src.chat_memory import (
    init_chat_table,
    save_message,
    load_messages,
    list_sessions,
    delete_session,
)
from src.monitoring import (
    start_monitoring_server,
    track_request,
    record_feedback,
    update_active_sessions,
    record_llm_model,
    get_average_response_time
)

# ---------------- Initialize ----------------
load_dotenv()
init_chat_table()

# ---------------- Start Prometheus Monitoring ----------------
MONITORING_PORT = 8000

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0

if not is_port_in_use(MONITORING_PORT):
    start_monitoring_server(MONITORING_PORT)
    print(f"✅ Monitoring server started on port {MONITORING_PORT}")
else:
    print(f"⚠️ Monitoring already running on port {MONITORING_PORT}")

# ---------------- LLM Setup ----------------
_groq = Groq(api_key=config.GROQ_API_KEY) if config.GROQ_API_KEY else None

def groq_generate(prompt: str, model_name: str = None) -> str:
    if _groq is None:
        return "LLM not configured. Please set GROQ_API_KEY in .env."
    model_name = model_name or config.GROQ_MODEL_NAME
    resp = _groq.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="Mindra RAG Chat", layout="centered")
st.title("💬 Mindra — Medical Assistant (Qdrant RAG Chat)")
st.markdown("Ask medical questions. The app retrieves from verified FAQ knowledge and answers contextually.")

# ---------------- Session State ----------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_selected" not in st.session_state:
    st.session_state.session_selected = "new"
if "last_selected_session" not in st.session_state:
    st.session_state.last_selected_session = "New Session"
if "delete_triggered" not in st.session_state:
    st.session_state.delete_triggered = False
# Track session start time for active sessions calculation
if "session_start_time" not in st.session_state:
    st.session_state.session_start_time = datetime.now()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")

    chat_memory_enabled = st.checkbox("💬 Enable Chat Memory", value=True)

    # --- Constant 'New Session' Button ---
    if st.button("🆕 New Session", type="primary", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.session_selected = "new"
        st.session_state.last_selected_session = "New Session"
        st.session_state.session_start_time = datetime.now()  # Reset session timer
        st.rerun()

    st.markdown("---")
    st.subheader("Chat History")

    sessions = list_sessions(limit=20)

    if sessions:
        st.markdown("""
        <style>
        .chat-history-scroll {
            max-height: 300px;
            overflow-y: auto;
            padding-right: 5px;
        }
        .chat-button {
            text-align: left;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="chat-history-scroll">', unsafe_allow_html=True)
        for session in sessions:
            messages = list(load_messages(session["session_id"]))
            if messages:
                first_msg = next((m for m in messages if m["role"] == "user"), messages[0])
                display_name = first_msg["content"][:30] + ("..." if len(first_msg["content"]) > 30 else "")
            else:
                display_name = f"Chat {session['last_update'].strftime('%m/%d %H:%M')}"

            cols = st.columns([4, 1], gap="small")
            with cols[0]:
                if st.button(
                    display_name,
                    key=f"load_{session['session_id']}",
                    use_container_width=True,
                    help=f"Last updated: {session['last_update'].strftime('%Y-%m-%d %H:%M')}",
                ):
                    st.session_state.session_id = session["session_id"]
                    st.session_state.messages = messages
                    st.session_state.session_selected = session["session_id"]
                    st.session_state.last_selected_session = display_name
                    st.session_state.session_start_time = datetime.now()  # Reset session timer
                    st.rerun()
            with cols[1]:
                if st.button("🗑️", key=f"delete_{session['session_id']}", help="Delete chat"):
                    delete_session(session["session_id"])
                    if st.session_state.session_id == session["session_id"]:
                        st.session_state.session_id = str(uuid.uuid4())
                        st.session_state.messages = []
                        st.session_state.session_selected = "new"
                        st.session_state.last_selected_session = "New Session"
                        st.session_state.session_start_time = datetime.now()
                    st.session_state.delete_triggered = True
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No chat history yet. Start a conversation!")

    st.markdown("---")
    st.subheader("Search Settings")

    qtype = st.selectbox("Question type (filter)", ["All", "information", "symptom", "treatment", "diagnosis"])
    qtype = None if qtype == "All" else qtype

    top_k = st.slider("Hybrid (RRF) candidates", 3, 20, value=config.DEFAULT_TOP_K)
    rerank_k = st.slider("Rerank top-k", 1, 10, value=config.DEFAULT_RERANK_K)
    llm_model = st.text_input("LLM model name", value=config.GROQ_MODEL_NAME)

    st.markdown("---")
    st.caption("Uses Qdrant hybrid collection with both dense and BM25 vectors.")

    # Display average response time
    st.write(f"Average response time: {get_average_response_time():.2f}s")

# ---------------- Chat Display ----------------
if st.session_state.delete_triggered and st.session_state.session_selected == "new":
    st.session_state.messages = []
    st.session_state.delete_triggered = False

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- Core Chat Logic ----------------
@track_request
def handle_user_query(user_query, qtype, top_k, rerank_k, llm_model):
    points = retriever.rrf_hybrid_search(query=user_query, qtype=qtype, limit=top_k)
    payloads = retriever.get_payloads_from_points(points)
    reranked = reranker.rerank(user_query, payloads, top_k=rerank_k)
    contexts = [p for p, _ in reranked]
    prompt = utils.build_prompt(user_query, contexts)
    return groq_generate(prompt, model_name=llm_model)

# ---------------- User Query Input ----------------
if user_query := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    if chat_memory_enabled:
        save_message(st.session_state.session_id, "user", user_query)

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            start = time.time()
            # Track the request with proper timing
            answer = handle_user_query(user_query, qtype, top_k, rerank_k, llm_model)
            elapsed = time.time() - start

            st.markdown(answer)
            st.caption(f"⏱️ Answer generated in {elapsed:.2f}s")

        st.session_state.messages.append({"role": "assistant", "content": answer})
        if chat_memory_enabled:
            save_message(st.session_state.session_id, "assistant", answer)

        # Update metrics - FIXED: Only record model once per session
        if len(st.session_state.messages) == 2:  # First user-assistant pair
            record_llm_model(llm_model)
        
        # FIXED: Update active sessions based on session activity, not message count
        session_age = (datetime.now() - st.session_state.session_start_time).total_seconds()
        if session_age < 900:  # 15 minutes of inactivity
            update_active_sessions(1)
        else:
            update_active_sessions(0)

        # Feedback Section
        st.write("### Was this answer helpful?")
        col1, col2 = st.columns(2)
        if col1.button("👍 Yes", key=f"pos_{st.session_state.session_id}_{len(st.session_state.messages)}"):
            record_feedback(True)
            st.success("Feedback recorded. Thank you!")
        if col2.button("👎 No", key=f"neg_{st.session_state.session_id}_{len(st.session_state.messages)}"):
            record_feedback(False)
            st.warning("Feedback recorded. Thank you!")

# ---------------- Sidebar System Status ----------------
st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
try:
    qdrant = retriever.get_qdrant_client()
    st.sidebar.write("Qdrant connected:", bool(qdrant))
except Exception as e:
    st.sidebar.error(f"Qdrant error: {e}")

st.sidebar.write("Embedding model:", config.EMBEDDING_MODEL_NAME)
st.sidebar.write("Reranker model:", config.RERANKER_MODEL)
st.sidebar.write("Groq configured:", bool(config.GROQ_API_KEY))