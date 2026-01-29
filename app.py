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
LOGO_PATH = "logo.png"          # repo root
DOCS_PATH = "docs"              # upload docs here
INDEX_PATH = "data/faiss_index" # cache on Streamlit Cloud

st.set_page_config(
    page_title="Humanio AI",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🤝",
    layout="wide",
)

# -----------------------------
# CHECKS
# -----------------------------
if not os.path.exists(LOGO_PATH):
    st.error("logo.png not found in repo root. Upload it as **logo.png**.")
    st.stop()

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found in Streamlit Secrets.")
    st.stop()

# -----------------------------
# SESSION STATE
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_seed" not in st.session_state:
    st.session_state.input_seed = 0  # used to reset form widget keys

def clear_chat():
    st.session_state.messages = []
    st.session_state.input_seed += 1  # forces a fresh input widget

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
# UI HEADER (CENTERED)
# -----------------------------
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

l1, c1, r1 = st.columns([5, 1, 5])
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

l2, c2, r2 = st.columns([5, 1.5, 5])
with c2:
    if st.button("🆕 New chat", use_container_width=True):
        clear_chat()
        st.rerun()

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.divider()

# -----------------------------
# WELCOME (ONLY WHEN EMPTY)
# -----------------------------
if len(st.session_state.messages) == 0:
    st.markdown(
        "<div style='text-align:center; opacity:0.75; margin-top: 6px;'>"
        "Hi 👋 I’m Humanio. Ask me anything about your HR policies, processes, or benefits."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# -----------------------------
# CHAT HISTORY
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# ANSWER LOGIC (CONVERSATIONAL)
# -----------------------------
def answer_question(user_q: str):
    # show user message
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # build/load KB
    try:
        with st.spinner("Thinking..."):
            vectorstore = get_vectorstore()
    except RateLimitError:
        msg = (
            "I’m getting rate-limited while building the knowledge base.\n\n"
            "Quick fixes:\n"
            "- Make sure your OpenAI account has billing/credits enabled\n"
            "- Try again in a minute\n"
            "- Reduce the number/size of documents in /docs\n"
        )
        with st.chat_message("assistant"):
            st.markdown(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        return

    if not vectorstore:
        msg = (
            "I don’t have any HR documents to learn from yet.\n\n"
            "Upload PDFs or .txt files into the **/docs** folder in GitHub, then refresh the app."
        )
        with st.chat_message("assistant"):
            st.markdown(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        return

    # stable retrieval
    retrieved = vectorstore.similarity_search(user_q, k=4)
    context = "\n\n---\n\n".join([d.page_content for d in retrieved])

    llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.4)

    prompt = f"""
You are Humanio AI, a friendly internal HR assistant.
Be conversational, helpful, and clear.
Use ONLY the provided context.
If the context doesn’t contain the answer, say so and suggest who to contact or what to check next.
Keep answers concise, but include steps/bullets where useful.

CONTEXT:
{context}

QUESTION:
{user_q}

ANSWER:
""".strip()

    try:
        resp = llm.invoke(prompt)
        answer = (resp.content or "").strip()
    except RateLimitError:
        answer = "I hit a rate limit while answering. Please try again in a minute."

    if not answer:
        answer = "I couldn’t generate a response. Please try again."

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# -----------------------------
# INPUT (FORM = CLEARS ON SUBMIT)
# -----------------------------
pad_l, mid, pad_r = st.columns([2.5, 5, 2.5])

with mid:
    with st.form(key=f"ask_form_{st.session_state.input_seed}", clear_on_submit=True):
        in_l, in_r = st.columns([8, 2])
        with in_l:
            q = st.text_input(
                label="Ask an HR question",
                placeholder="Ask an HR question…",
                label_visibility="collapsed",
            )
        with in_r:
            sent = st.form_submit_button("Send", use_container_width=True)

    if sent:
        q = (q or "").strip()
        if not q:
            st.warning("Type a question first.")
        else:
            answer_question(q)
            st.rerun()
