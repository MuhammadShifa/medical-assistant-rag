import time
import threading
from prometheus_client import Counter, Gauge, Histogram, start_http_server, REGISTRY

# -----------------------------
# Global state to track initialization
# -----------------------------

_metrics_initialized = False
_metrics_lock = threading.Lock()
_server_started = False
_server_lock = threading.Lock()

# Global metric references
REQUEST_COUNT = None
RESPONSE_TIME = None
USER_FEEDBACK = None
ACTIVE_SESSIONS = None
LLM_MODEL_USED = None

# For calculating average response time
_response_data = {
    'total_time': 0.0,
    'total_requests': 0
}
_response_lock = threading.Lock()

# -----------------------------
# Initialization Function
# -----------------------------

def _initialize_metrics():
    """Initialize metrics only once."""
    global _metrics_initialized, REQUEST_COUNT, RESPONSE_TIME, USER_FEEDBACK, ACTIVE_SESSIONS, LLM_MODEL_USED
    
    with _metrics_lock:
        if not _metrics_initialized:
            try:
                # Check if metrics already exist in registry and remove them
                for metric_name in ['rag_requests_total', 'rag_response_time_seconds', 
                                  'rag_user_feedback_total', 'rag_active_sessions', 
                                  'rag_llm_model_used_total']:
                    if metric_name in REGISTRY._names_to_collectors:
                        REGISTRY.unregister(REGISTRY._names_to_collectors[metric_name])
                
                # Initialize metrics
                REQUEST_COUNT = Counter(
                    "rag_requests_total", 
                    "Total number of queries processed by the RAG system"
                )

                RESPONSE_TIME = Histogram(
                    "rag_response_time_seconds",
                    "Response time for RAG responses in seconds",
                    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
                )

                USER_FEEDBACK = Counter(
                    "rag_user_feedback_total", 
                    "User feedback counts", 
                    ["feedback_type"]
                )

                ACTIVE_SESSIONS = Gauge(
                    "rag_active_sessions", 
                    "Number of currently active chat sessions"
                )

                LLM_MODEL_USED = Counter(
                    "rag_llm_model_used_total", 
                    "Number of queries served per LLM model", 
                    ["model_name"]
                )

                _metrics_initialized = True
                print("✅ Metrics initialized successfully")
                
            except Exception as e:
                print(f"❌ Error initializing metrics: {e}")

# -----------------------------
# Helper Functions
# -----------------------------

def start_monitoring_server(port=8000):
    """Start Prometheus metrics endpoint."""
    global _server_started
    
    # Initialize metrics first
    _initialize_metrics()
    
    with _server_lock:
        if not _server_started:
            try:
                start_http_server(port, addr="0.0.0.0")
                _server_started = True
                print(f"✅ Prometheus monitoring server started on port {port}")
            except Exception as e:
                print(f"❌ Failed to start monitoring server: {e}")

def track_request(func):
    """Decorator to measure latency and count requests."""
    
    def wrapper(*args, **kwargs):
        # Ensure metrics are initialized
        _initialize_metrics()
        
        # Increment request count
        if REQUEST_COUNT:
            REQUEST_COUNT.inc()
        
        # Measure response time
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            
            # Record in histogram
            if RESPONSE_TIME:
                RESPONSE_TIME.observe(duration)
            
            # Update average response time calculation
            with _response_lock:
                _response_data['total_time'] += duration
                _response_data['total_requests'] += 1
    
    return wrapper

def record_feedback(is_positive: bool):
    """Record user feedback (positive/negative)."""
    _initialize_metrics()
    
    try:
        if USER_FEEDBACK:
            feedback_type = "positive" if is_positive else "negative"
            USER_FEEDBACK.labels(feedback_type=feedback_type).inc()
            print(f"✅ Feedback recorded: {feedback_type}")
    except Exception as e:
        print(f"❌ Error recording feedback: {e}")

def update_active_sessions(count: int):
    """Set active sessions gauge."""
    _initialize_metrics()
    
    try:
        if ACTIVE_SESSIONS:
            ACTIVE_SESSIONS.set(count)
            print(f"✅ Active sessions updated: {count}")
    except Exception as e:
        print(f"❌ Error updating active sessions: {e}")

def record_llm_model(model_name: str):
    """Record which LLM model is used."""
    _initialize_metrics()
    
    try:
        if LLM_MODEL_USED and model_name and model_name.strip():
            LLM_MODEL_USED.labels(model_name=model_name).inc()
            print(f"✅ LLM model recorded: {model_name}")
    except Exception as e:
        print(f"❌ Error recording LLM model: {e}")

def get_average_response_time():
    """Return the average response time so far."""
    with _response_lock:
        if _response_data['total_requests'] == 0:
            return 0.0
        return _response_data['total_time'] / _response_data['total_requests']

def get_metrics_summary():
    """Get a summary of current metrics for debugging."""
    active_sessions_value = 0
    if ACTIVE_SESSIONS and hasattr(ACTIVE_SESSIONS, '_value'):
        active_sessions_value = ACTIVE_SESSIONS._value.get()
    
    return {
        'total_requests': _response_data['total_requests'],
        'average_response_time': get_average_response_time(),
        'active_sessions': active_sessions_value
    }