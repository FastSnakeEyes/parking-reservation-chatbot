"""Token counting for history trimming and summarization.

"tiktoken"    -> exact counts for OpenAI models (via ChatOpenAI, which uses tiktoken locally;
                 no API call is made).
"approximate" -> LangChain's fast heuristic, no extra dependency.
"""

from collections.abc import Callable, Sequence
from functools import lru_cache

from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_openai import ChatOpenAI

from app.config import settings

TokenCounter = Callable[[Sequence[BaseMessage]], int]


@lru_cache
def get_token_counter() -> TokenCounter:
    if settings.token_counter == "tiktoken":  # noqa: S105 (not a secret)
        counter_model = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key.get_secret_value(),
        )
        return counter_model.get_num_tokens_from_messages
    return count_tokens_approximately


def count_tokens(messages: Sequence[BaseMessage]) -> int:
    return get_token_counter()(list(messages))
