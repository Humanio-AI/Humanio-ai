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
# CONFIG
# -----------------------------
LOGO_PATH = "logo.png"
DOCS_PATH = "docs"
INDEX_PATH = "data/faiss_index"

st.set_page_config(
    page_title="Flatpay",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🤝",
    layout="wide",
)

# -----------------------------
# HIDE STREAMLIT HEADER / ICONS
# -----------------------------
st.markdown(
    """
    <style>
      header[data-testid="stHeader"] {display:none;}
      div[data-testid="stToolbar"] {display:none;}
      .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# SAFETY CHECKS
# -----------------------------
if not os.path.exists(LOGO_PATH):
    st.error("logo.png not found in repo root.")
    st.stop()

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY missing in Streamlit Secrets.")
    st.stop()

# -----------------------------
# SESSION
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_seed" not in st.session_state:
    st.session_state.input_seed = 0

def clear_chat():
    st.session_state.messages = []
    st.session_state.input_seed += 1

# -----------------------------
# RAG HELPERS
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
        chunk_size=64,
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
# CENTERED LAYOUT
# -----------------------------
pad_l, center, pad_r = st.columns([2, 5, 2])

with center:

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ✅ PERFECTLY CENTERED LOGO
    st.image(LOGO_PATH, width=120)

    # ✅ CENTER TITLE
    st.markdown(
        """
        <div style="text-align:center;">
            <h1>Flatpay People Team</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ✅ CENTER BUTTON (not too wide)
    bl, bm, br = st.columns([2, 3, 2])
    with bm:
        if st.button("New chat", use_container_width=True):
            clear_chat()
            st.rerun()

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.divider()

    if len(st.session_state.messages) == 0:
        st.markdown(
            "<div style='text-align:center; opacity:0.75;'>"
            "I am here to support with your HR queries"
            "</div>",
            unsafe_allow_html=True,
        )

    # -----------------------------
    # CHAT HISTORY
    # -----------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # -----------------------------
    # ANSWER LOGIC
    # -----------------------------
    def answer_question(user_q: str):
        st.session_state.messages.append({"role": "user", "content": user_q})

        with st.chat_message("user"):
            st.markdown(user_q)

        try:
            with st.spinner("Analyzing..."):
                vectorstore = get_vectorstore()
        except RateLimitError:
            msg = "OpenAI rate limit hit. Try again shortly."
            with st.chat_message("assistant"):
                st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            return

        if not vectorstore:
            msg = "No HR docs found. Upload PDFs/TXT to /docs folder."
            with st.chat_message("assistant"):
                st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            return

        retrieved = vectorstore.similarity_search(user_q, k=4)
        context = "\n\n---\n\n".join([d.page_content for d in retrieved])

        llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.4)

        prompt = f"""
You are a friendly HR assistant for Flatpay.
Be conversational and helpful.
Use ONLY the context below. If the answer isn't there, say so.

CONTEXT:
{context}

QUESTION:
{user_q}

ANSWER:
""".strip()

        resp = llm.invoke(prompt)
        answer = (resp.content or "").strip()

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    # -----------------------------
    # INPUT BOX (NARROWER)
    # -----------------------------
    il, im, ir = st.columns([1, 6, 1])

    with im:
        with st.form(key=f"ask_form_{st.session_state.input_seed}", clear_on_submit=True):
            q = st.text_input(
                label="Ask an HR question",
                placeholder="Ask an HR question…",
                label_visibility="collapsed",
            )
            sent = st.form_submit_button("Send", use_container_width=True)

        if sent:
            q = (q or "").strip()
            if q:
                answer_question(q)
                st.rerun()
