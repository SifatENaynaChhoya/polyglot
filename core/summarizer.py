from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os 

# Was 3000. mistral-small-latest takes 32k tokens, so 3000 chars per chunk meant
# a 90-minute meeting became 25+ API calls instead of 4 — which is exactly where
# the free-tier rate limits bite. Bigger chunks also mean each partial summary
# sees more context, so it summarizes better.
MAX_CHARS = 20000


def get_llm():
    return ChatMistralAI(model = "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"),temperature=0.3)


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = MAX_CHARS,
        chunk_overlap = 200
    )

    return splitter.split_text(transcript)


def fold_summaries(summaries : list, combined_chain) -> str:
    """Combine the partial summaries in rounds.

    The original code joined every partial and sent it in one call. That works
    for a short meeting and overflows for a long one — the map step was
    budgeted but the combine step wasn't. Grouping the partials so each combine
    call stays under MAX_CHARS removes the ceiling.
    """
    level = summaries

    while True:
        batches = []
        batch = []
        size = 0

        for item in level:
            if batch and size + len(item) > MAX_CHARS:
                batches.append(batch)
                batch = []
                size = 0
            batch.append(item)
            size += len(item)

        if batch:
            batches.append(batch)

        # Everything fits in one call — this is the final combine pass.
        if len(batches) == 1:
            return combined_chain.invoke("\n\n".join(batches[0]))

        # A single partial is already oversized; trim rather than loop forever.
        if len(batches) == len(level):
            return combined_chain.invoke("\n\n".join(level)[:MAX_CHARS])

        next_level = []
        for b in batches:
            if len(b) == 1:
                next_level.append(b[0])
            else:
                next_level.append(combined_chain.invoke("\n\n".join(b)))

        level = next_level


def summarize(transcript : str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "Summarize this portion of a meeting transcript concisely, as "
                   "bullet points. This is one segment of a longer recording and "
                   "may start or end mid-sentence. Keep speaker attributions and "
                   "any numbers, dates or names exactly as stated."),
        ("human", "{text}"),
    ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing segment {i+1}/{len(chunks)}...")
        chunk_summaries.append(map_chain.invoke({"text" : chunk}))

    combined_prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "You are an expert meeting summarizer. Combine these partial summaries "
            "into one final professional meeting summary in bullet points. "
            "Group by topic rather than by segment, and merge repeated points "
            "instead of listing them twice — the segments overlap.",
        ),
        ("human", "{text}"),
    ]
    )

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | combined_prompt | llm | StrOutputParser()
    )

    return fold_summaries(chunk_summaries, combined_chain)

def generate_title(transcript : str) -> str:
    llm = get_llm()

    

    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | 
        ChatPromptTemplate.from_messages([
             (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(max 8 words). Only return the title, nothing else.",
            ),
            ("human", "{text}"),
        ])
        | llm
        |StrOutputParser()
    )

    return title_chain.invoke(transcript[:2000])