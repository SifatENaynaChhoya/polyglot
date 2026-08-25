import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import build_vector_store, load_vector_store, get_retriever


ANSWER_PROMPT = """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.
If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."
Always be concise and precise. The transcript has no speaker labels, so never
attribute a statement to a named person unless that name appears in the context.

Context from meeting transcript:
{context}"""


def get_llm(temperature : float = 0.0):
    # temperature 0 for answering: the task is to report what the context says,
    # not to be creative. 0.3 invites drift away from the source.
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=temperature,
    )


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def normalize_query(question : str) -> str:
    """Rewrite the question in English before it is embedded.

    Transcripts are always English (Sarvam translates), so matching in English
    removes the cross-lingual burden from MiniLM. This also covers romanized
    Bangla like "ki decision hoyeche", which embedding models handle worse than
    native script.
    """
    chain = (
        ChatPromptTemplate.from_messages([
            ("system",
             "Rewrite the user's question in clear English. The input may be "
             "Bengali or Hindi script, romanized Bengali/Hindi, code-mixed, or "
             "already English. Preserve the exact meaning and any names or "
             "numbers. Return only the rewritten question, nothing else."),
            ("human", "{text}"),
        ])
        | get_llm(temperature=0.0)
        | StrOutputParser()
    )
    return chain.invoke({"text" : question}).strip()


def _make_chain(retriever):
    """Shared by build_rag_chain and load_rag_chain.

    Those two were identical apart from where the vector store came from,
    including a copy of the whole prompt. Editing one and forgetting the other
    is a bug waiting to happen.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_PROMPT),
        ("human", "{question}"),
    ])

    return (
        {
            "context" : retriever | RunnableLambda(format_docs),
            "question" : RunnablePassthrough(),
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )


def build_rag_chain(transcript : str, source : str):
    """Index the transcript, then return a chain scoped to that source."""
    vector_store = build_vector_store(transcript, source)
    return _make_chain(get_retriever(vector_store, source, k=5))


def load_rag_chain(source : str):
    """Reuse an already-indexed transcript — no re-embedding."""
    vector_store = load_vector_store()
    return _make_chain(get_retriever(vector_store, source, k=5))


def ask_question(rag_chain, question : str) -> str:
    english = normalize_query(question)

    # Printing both matters for debugging: if an answer is wrong, this tells you
    # instantly whether normalization or retrieval was at fault.
    if english.lower() != question.strip().lower():
        print(f"Question : {question}  ->  {english}")
    else:
        print(f"Question : {question}")

    answer = rag_chain.invoke(english)
    print(f"Answer : {answer}")
    return answer