import os
import time
import shutil
import streamlit as st
from openai import RateLimitError

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# Config
# -----------------------------
LOGO_PATH = "logo.png"
DOCS_PATH = "docs"
INDEX_PATH = "data/faiss_index"

# Must be first Streamlit call:
st.set_page_config(
    page_title="Humanio AI",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🤝",
    layout="wide",
)

# -----------------------------
# Basic checks (never blank)
# -----------------------------
if not os.path.exists(LOGO_PATH):
    st.error("logo.png not found in repo root. Upload it as 'logo.png'.")
    st.stop()

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found in Streamlit Secrets.")
    st.stop()

# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "draft" not in st.session_state:
    st.session_state.draft = ""

def clear_chat():
    st.session_state.messages = []
    st.session_state.draft = ""

# -----------------------------
# RAG helpers
# -----------------------------
def ensure_dirs():
    os.makedirs(DOCS_PATH, exist_ok=True)
    os.makedirs("data", exist_ok=True)

def delete_index():
    if os.path.exists(INDEX_PATH):
        shutil.rmtree(INDEX_PATH, ignore_errors=True)

def load_documents():
    ensure_dirs()
    docs = []
    for root, _, files in os.walk(DOCS_PATH):
        for f in files:
            fp = os.path.join(root, f)
            if f.lower().endswith(".pdf"):
                docs.extend(PyPDFLoader(fp).load())
            elif f.lower().endswith(".txt"):
                docs.extend(TextLoader(fp, encoding="utf-8").load())
    return docs

def load_vectorstore_if_exists():
    embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

def build_vectorstore_with_backoff(max_retries=5):
    docs = load_documents()
    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        model="text-embedding-3-small",
        chunk_size=64,  # smaller batches reduces rate spikes
    )

    delay = 2
    for attempt in range(1, max_retries + 1):
        try:
            vs = FAISS.from_documents(chunks, embeddings)
            os.makedirs(INDEX_PATH, exist_ok=True)
            vs.save_local(INDEX_PATH)
            return vs
        except RateLimitError:
            if attempt == max_retries:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30)

def get_vectorstore():
    vs = load_vectorstore_if_exists()
    if vs:
        return vs

    delete_index()
    return build_vectorstore_with_backoff()

# -----------------------------
# UI (Centered logo/title/button)
# -----------------------------
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

l1, c1, r1 = st.columns([4, 1, 4])
with c1:
    st.image(LOGO_PATH, width=90)

st.markdown(
    """
    <div style="text-align:center;">
        <h1 style="margin-bottom:0.2rem;">Humanio AI</h1>
        <div style="opacity:0.8;">Internal HR Assistant</div>
    </div>
    """,
    unsafe_allow_html=True,
)

l2, c2, r2 = st.columns([4, 2, 4])
with c2:
    if st.button("🆕 New chat"):
        clear_chat()
        st.rerun()

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.divider()

# -----------------------------
# Render chat messages
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# Narrow input (custom, not st.chat_input)
# -----------------------------
pad_l, mid, pad_r = st.columns([1, 6, 1])

with mid:
    in_l, in_r = st.columns([10, 2])
    with in_l:
        st.session_state.draft = st.text_input(
            label="Ask an HR question",
            value=st.session_state.draft,
            placeholder="Ask an HR question…",
            label_visibility="collapsed",
        )
    with in_r:
        send = st.button("Send", use_container_width=True)

def run_question(user_q: str):
    st.session_state.messages.append({"role": "user", "content": user_q})

    with st.chat_message("user"):
        st.markdown(user_q)

    try:
        with st.spinner("Thinking..."):
            vectorstore = get_vectorstore()

        if not vectorstore:
            answer = (
                "I don’t have any HR documents yet.\n\n"
                "Upload PDFs/TXT into the **/docs** folder in GitHub, then refresh."
            )
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            return

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        retrieved = retriever.get_relevant_documents(user_q)
        context = "\n\n---\n\n".join([d.page_content for d in retrieved])

        llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.2)

        prompt = f"""
You are Humanio AI, an internal HR assistant.
Answer ONLY using the provided context.
If the context does not contain the answer, say you don't know.

CONTEXT:
{context}

QUESTION:
{user_q}

ANSWER:
""".strip()

        resp = llm.invoke(prompt)
        answer = resp.content

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    except RateLimitError:
        msg = (
            "I hit an OpenAI rate limit while building the knowledge base.\n\n"
            "- Check your OpenAI account has billing/credits enabled\n"
            "- Try again in a couple of minutes\n"
            "- Reduce the number/size of documents in /docs\n"
        )
        with st.chat_message("assistant"):
            st.markdown(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
    except Exception as e:
        st.error(f"App error: {type(e).__name__}: {e}")

# Send handling
if send and st.session_state.draft.strip():
    q = st.session_state.draft.strip()
    st.session_state.draft = ""  # clear input
    run_question(q)
    st.rerun()
