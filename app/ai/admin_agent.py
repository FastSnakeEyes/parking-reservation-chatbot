"""Stage 2: the "second agent" that drafts the administrator's reservation
approval request. The actual approve/reject exchange with the administrator
happens over the REST API (app/api/routers/admin.py) and the Streamlit
console (ui/admin_app.py); this module only turns a reservation's details
into a short, human-readable notification.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

DRAFT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You draft short reservation-approval requests for a parking "
            "operator's administrator to review. State only the facts given "
            "below; never invent or infer a detail that isn't listed. "
            "Two to four plain sentences, no greeting or sign-off.",
        ),
        ("human", "{details}"),
    ]
)


async def draft_notification(llm: ChatOpenAI, details: dict[str, str]) -> str:
    chain = DRAFT_PROMPT | llm | StrOutputParser()
    lines = "\n".join(f"{key}: {value}" for key, value in details.items())

    return await chain.ainvoke({"details": lines})
