#Actionableitems , decision , questions 

#Actionableitems , decision , questions 

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os 

# mistral-small-latest has a 32k token window, and that budget covers the system
# prompt + the transcript we send + the model's reply. ~20000 chars is roughly
# 6k tokens of transcript, which leaves plenty of headroom. Meeting transcripts
# tokenize worse than clean prose (Hinglish, transliteration, filler words), so
# assume ~3.3 chars per token rather than the usual 4.
MAX_CHARS = 20000


def get_llm():
    return ChatMistralAI(model = "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"),temperature=0.2)



def build_chain(system_prompt : str):
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )


def split_transcript(transcript : str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = MAX_CHARS,
        chunk_overlap = 500
    )

    return splitter.split_text(transcript)


def merge_partials(partials : list, merge_prompt : str) -> str:
    """Fold the per-chunk extractions down to one list.

    Folded in rounds rather than one big call, because with enough chunks the
    concatenated partials would overflow the window themselves — the same bug
    as before, just moved later in the pipeline.
    """
    merge_chain = build_chain(merge_prompt)
    level = partials

    while len(level) > 1:
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

        # No grouping possible (a single partial is already oversized) — merge
        # once against a trimmed input rather than looping forever.
        if len(batches) == len(level):
            return merge_chain.invoke("\n\n".join(level)[:MAX_CHARS])

        next_level = []
        for b in batches:
            if len(b) == 1:
                next_level.append(b[0])
            else:
                next_level.append(merge_chain.invoke("\n\n".join(b)))

        level = next_level

    return level[0]


def run_extraction(transcript : str, map_prompt : str, merge_prompt : str, label : str) -> str:
    chunks = split_transcript(transcript)
    map_chain = build_chain(map_prompt)

    # Short transcript — single call, no merge pass to degrade the output.
    if len(chunks) == 1:
        return map_chain.invoke(chunks[0])

    partials = []
    for i, chunk in enumerate(chunks):
        print(f"{label} — segment {i+1}/{len(chunks)}...")
        partials.append(map_chain.invoke(chunk))

    return merge_partials(partials, merge_prompt)


# Chunks overlap by 500 chars and speakers repeat themselves, so the same item
# will legitimately appear in several partials. Without an explicit merge
# instruction the final list comes back with duplicates.
MERGE_RULES = (
    "You are merging partial extractions from consecutive segments of ONE "
    "meeting transcript.\n"
    "- Merge items that mean the same thing even if worded differently; keep "
    "the more specific version.\n"
    "- Segments overlap, so near-duplicates are expected. Never list an item twice.\n"
    "- Keep the order in which items first appeared.\n"
    "- Do not invent anything that is not in the partials.\n"
    "- Ignore any partial that says NONE.\n"
    "- Return one clean numbered list and nothing else.\n"
)


def extract_action_items(transcript:str)->str:
    return run_extraction(
        transcript,
         "You are an expert meeting analyst. This is ONE SEGMENT of a longer "
        "meeting transcript, so it may start or end mid-sentence — that is "
        "expected. From the segment, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found in this segment, reply exactly: NONE",

        MERGE_RULES +
        "Each item keeps its Task / Owner / Deadline fields. If one partial "
        "names an owner or deadline where another says 'Not specified', use the "
        "named one. If every partial says NONE, reply 'No action items found.'",

        "Extracting action items",
    )


def extract_key_decisions(transcript: str) -> str:
    return run_extraction(
        transcript,
        "You are an expert meeting analyst. This is ONE SEGMENT of a longer "
        "meeting transcript and may start or end mid-sentence. From the segment, "
        "extract all key decisions made — decisions actually taken, not options "
        "still under discussion. Format as a numbered list. "
        "If none found in this segment, reply exactly: NONE",

        MERGE_RULES +
        "If an earlier partial records a decision that a later partial reverses "
        "or supersedes, keep only the final one. If every partial says NONE, "
        "reply 'No key decisions found.'",

        "Extracting decisions",
    )


def extract_questions(transcript: str) -> str:
    return run_extraction(
        transcript,
        "This is ONE SEGMENT of a longer meeting transcript and may start or "
        "end mid-sentence. From the segment, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found in this segment, reply exactly: NONE",

        MERGE_RULES +
        "Drop any question from an earlier partial that a later partial shows "
        "was answered during the meeting — only genuinely open items survive. "
        "If every partial says NONE, reply 'No open questions found.'",

        "Extracting open questions",
    )