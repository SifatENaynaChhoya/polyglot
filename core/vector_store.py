import hashlib

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"

# Stays English-only. Sarvam's endpoint translates, so every transcript reaching
# this file is already English. The user's question gets rewritten to English in
# qa_chain.py before embedding, which also handles romanized Bangla ("ki
# decision hoyeche") that no embedding model reads well.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# 900/100 is a guess, not a fix — 500/50 at k=4 gave the LLM only ~2000 chars,
# which felt thin for answers spanning a minute of speech. Revert to 500/50 if
# your testing says otherwise.
#
# WARNING: if you change EMBEDDING_MODEL or these sizes, delete the vector_db
# folder first. Old vectors come from a different vector space and silently
# make retrieval worse instead of erroring.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 100

_embeddings = None


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
        _embeddings = HuggingFaceEmbeddings(
            model_name = EMBEDDING_MODEL,
            model_kwargs = {"device" : 'cpu'}
        )

    return _embeddings


def source_id(source : str) -> str:
    """Short stable id for a video URL or file path. Every call site must
    compute this the same way or the metadata filter matches nothing."""
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def build_vector_store(transcript : str, source : str) -> Chroma:
    print("Building vector Store")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP
    )
    chunks = splitter.split_text(transcript)

    sid = source_id(source)
    store = load_vector_store()

    # from_documents() appends, so re-running on the same video used to duplicate
    # every chunk. Drop this source's old rows first to make re-runs safe.
    stale = store.get(where = {"source_id" : sid}, include = [])
    if stale["ids"]:
        store.delete(ids = stale["ids"])

    docs = [
        # chunk_index is a character offset, not a time. Add start_seconds here
        # once transcriber.py returns timed segments.
        Document(page_content = chunk, metadata = {"chunk_index" : i, "source_id" : sid})
        for i, chunk in enumerate(chunks)
    ]

    store.add_documents(documents = docs, ids = [f"{sid}:{i}" for i in range(len(docs))])
    print(f"Indexed {len(docs)} chunk(s).")

    return store


def load_vector_store() -> Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name = COLLECTION_NAME,
        embedding_function = embeddings,
        persist_directory = CHROMA_DIR
    )

    return vector_store


def get_retriever(vector_store : Chroma, source : str, k : int = 5):
    return vector_store.as_retriever(
        search_type = 'similarity',
        # Without this filter, k=5 pulls the most similar chunks across every
        # video ever indexed, not just this one.
        search_kwargs = {"k" : k, "filter" : {"source_id" : source_id(source)}}
    )