"""The customer conversation (app/ai/graph.py, app/api/routers/chat.py)."""

# Prefix of chat threads in the shared checkpointer (app/ai/chat_cleanup.py finds them by it).
CHAT_THREAD_PREFIX = "chat:"

CHAT_REQUEST_TIMEOUT_SECONDS = 120

# Per-thread Redis lock shared by chat turns and admin resumes.
THREAD_LOCK_TIMEOUT_SECONDS = 180  # auto-released if a holder dies
CHAT_LOCK_WAIT_SECONDS = 1  # a customer's concurrent message is refused quickly
ADMIN_LOCK_WAIT_SECONDS = 10  # an admin decision waits out an in-flight chat turn
