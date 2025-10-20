import time

import streamlit as st
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# -----------------------------
# Prometheus Metric Definitions
# -----------------------------

# Streamlit-safe initialization
if "REQUEST_COUNT" not in st.session_state:
    st.session_state.REQUEST_COUNT = Counter(
        "rag_request_count_total", "Total number of queries processed by the RAG system"
    )
REQUEST_COUNT = st.session_state.REQUEST_COUNT

if "RESPONSE_TIME" not in st.session_state:
    st.session_state.RESPONSE_TIME = Histogram(
        "rag_response_time_seconds",
        "Response time for RAG responses (seconds)",
        buckets=[0.5, 1, 2, 5, 10, 20],
    )
RESPONSE_TIME = st.session_state.RESPONSE_TIME

if "USER_FEEDBACK" not in st.session_state:
    st.session_state.USER_FEEDBACK = Counter(
        "rag_user_feedback_total", "User feedback counts", ["rating"]
    )
USER_FEEDBACK = st.session_state.USER_FEEDBACK

if "ACTIVE_SESSIONS" not in st.session_state:
    st.session_state.ACTIVE_SESSIONS = Gauge(
        "rag_active_sessions", "Number of currently active chat sessions"
    )
ACTIVE_SESSIONS = st.session_state.ACTIVE_SESSIONS

if "LLM_MODEL_USED" not in st.session_state:
    st.session_state.LLM_MODEL_USED = Counter(
        "rag_llm_model_used_total", "Number of queries served per LLM model", ["model"]
    )
LLM_MODEL_USED = st.session_state.LLM_MODEL_USED

if "RESPONSE_TIME_SUM" not in st.session_state:
    st.session_state.RESPONSE_TIME_SUM = 0.0
if "RESPONSE_COUNT" not in st.session_state:
    st.session_state.RESPONSE_COUNT = 0

# -----------------------------
# Helper Functions
# -----------------------------


def start_monitoring_server(port=8000):
    """Start Prometheus metrics endpoint (Streamlit-safe)."""
    if "PROMETHEUS_STARTED" not in st.session_state:
        start_http_server(port, addr="0.0.0.0")
        st.session_state.PROMETHEUS_STARTED = True
        print(f"✅ Prometheus monitoring server started on port {port}")


def track_request(func):
    """Decorator to measure latency and count requests."""

    def wrapper(*args, **kwargs):
        REQUEST_COUNT.inc()
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        RESPONSE_TIME.observe(duration)

        # Update average response time
        st.session_state.RESPONSE_TIME_SUM += duration
        st.session_state.RESPONSE_COUNT += 1
        return result

    return wrapper


def record_feedback(is_positive: bool):
    """Record user feedback (positive/negative)."""
    label = "positive" if is_positive else "negative"
    USER_FEEDBACK.labels(rating=label).inc()


def update_active_sessions(count: int):
    """Set active sessions gauge."""
    ACTIVE_SESSIONS.set(count)


def record_llm_model(model_name: str):
    """Record which LLM model is used."""
    LLM_MODEL_USED.labels(model=model_name).inc()


def get_average_response_time():
    """Return the average response time so far."""
    if st.session_state.RESPONSE_COUNT == 0:
        return 0.0
    return st.session_state.RESPONSE_TIME_SUM / st.session_state.RESPONSE_COUNT
