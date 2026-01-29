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
LOGO_PATH = "logo.png"          # must be in repo root
DOCS_PATH = "docs"              # upload PDFs/TXT here via GitHub
INDEX_PATH = "data/faiss_index" # cache on Streamlit Cloud

st.set_page_config(
    page_title="Humanio AI",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🤝",
    layout="wide",
)

# -----------------------------
# BASIC CHECKS
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
if "draft" not in st.session_state:
    st.session_state.draft = ""
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False

def clear_chat():
    st.session_state.messages = []
    st.session_state.draft = ""
    # keep kb_ready as-is (new chat shouldn't rebuild KB)

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
# HEADER UI (CENTERED)
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
# FRIENDLY WELCOME (only when empty chat)
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
# RENDER CHAT HISTORY
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# ANSWER LOGIC (conversational)
# -----------------------------
def run_question(user_q: str):
    # User message
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # Load/build KB
    try:
        with st.spinner("Thinking..."):
            vectorstore = get_vectorstore()
    except RateLimitError:
        msg = (
            "I’m getting rate-limited while building the knowledge base.\n\n"
            "A few quick fixes:\n"
            "- Make sure your OpenAI account has billing/credits enabled\n"
            "- Try again in a minute\n"
            "- Reduce the number/size of documents in /docs\n"
        )
        with st.chat_message("assistant"):
            st.markdown(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        return
    except Exception as e:
        msg = f"Something went wrong while preparing the knowledge base: **{type(e).__name__}** — {e}"
        with st.chat_message("assistant"):
            st.markdown(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        return

    # No docs yet
    if not vectorstore:
        msg = (
            "I don’t have any HR documents to learn from yet.\n\n"
            "**To fix:** upload PDFs or .txt files into the **/docs** folder in GitHub, then refresh this page."
        )
        with st.chat_message("assistant"):
            st.markdown(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        return

    # Retrieve (stable across versions)
    retrieved = vectorstore.similarity_search(user_q, k=4)
    context = "\n\n---\n\n".join([d.page_content for d in retrieved])

    llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.4)

    system_style = """
You are Humanio AI, a friendly internal HR assistant.
Be conversational and helpful.
Use ONLY the provided context to answer.
If the context doesn't contain the answer, say so clearly and suggest who to contact or what policy to check.
Keep answers concise but complete. Use bullet points when it helps.
""".strip()

    prompt = f"""
{system_style}

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
    except Exception as e:
        answer = f"Something went wrong while answering: **{type(e).__name__}** — {e}"

    if not answer:
        answer = "I couldn’t generate a response. Please try again."

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# -----------------------------
# NARROW INPUT ROW (TEXT CLEARS ON SEND + NEW CHAT)
# -----------------------------
pad_l, mid, pad_r = st.columns([2.5, 5, 2.5])

with mid:
    input_l, input_r = st.columns([8, 2])

    with input_l:
        # key is important: lets us clear it reliably
        st.text_input(
            label="Ask an HR question",
            value=st.session_state.draft,
            placeholder="Ask an HR question…",
            label_visibility="collapsed",
            key="draft_input",
        )

    with input_r:
        send = st.button("Send", use_container_width=True)

# Sync the typed value into state
st.session_state.draft = st.session_state.get("draft_input", "")

if send:
    q = (st.session_state.draft or "").strip()
    if not q:
        st.warning("Type a question first.")
    else:
        # Clear input immediately (so it disappears)
        st.session_state.draft = ""
        st.session_state["draft_input"] = ""
        run_question(q)
        st.rerun()
