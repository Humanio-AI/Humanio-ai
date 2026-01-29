import os
import shutil
import streamlit as st
from PIL import Image

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# App config + branding
# -----------------------------
LOGO_PATH = "logo.png"  # <- put logo.png in your repo root
DOCS_PATH = "docs"
INDEX_PATH = "data/faiss_index"

page_icon = "🤝"
if os.path.exists(LOGO_PATH):
    try:
        page_icon = Image.open(LOGO_PATH)
    except Exception:
        pass

st.set_page_config(page_title="Humanio AI", page_icon=page_icon, layout="wide")


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "kb_status" not in st.session_state:
        st.session_state.kb_status = ""


init_state()


# -----------------------------
# Secrets
# -----------------------------
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found in Streamlit Secrets.")
    st.stop()


# -----------------------------
# Helpers
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


def build_vectorstore():
    embeddings = OpenAIEmbeddings(api_key=api_key)

    docs = load_documents()
    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    vs = FAISS.from_documents(chunks, embeddings)
    os.makedirs(INDEX_PATH, exist_ok=True)
    vs.save_local(INDEX_PATH)
    return vs


def load_vectorstore_if_exists():
    embeddings = OpenAIEmbeddings(api_key=api_key)
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return None


def save_uploaded_files(uploaded_files):
    ensure_dirs()
    saved = []
    for uf in uploaded_files:
        name = uf.name
        lower = name.lower()
        if not (lower.endswith(".pdf") or lower.endswith(".txt")):
            continue
        out_path = os.path.join(DOCS_PATH, name)
        with open(out_path, "wb") as f:
            f.write(uf.getbuffer())
        saved.append(name)
    return saved


def new_chat():
    st.session_state.messages = []


# -----------------------------
# Header layout (no sidebar)
# -----------------------------
top_left, top_mid, top_right = st.columns([1, 6, 1])

with top_left:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=54)

with top_mid:
    st.markdown(
        """
        <div style="text-align:center; margin-top: 6px;">
            <h1 style="margin-bottom: 0.2rem;">Humanio AI</h1>
            <div style="opacity:0.75; font-size: 1.05rem;">Internal HR Assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    if st.button("🆕 New chat", use_container_width=True):
        new_chat()
        st.rerun()

st.divider()


# -----------------------------
# Knowledge base controls (main page)
# -----------------------------
with st.expander("📚 Knowledge base (upload HR docs + rebuild)", expanded=False):
    st.write("Upload HR documents (PDF/TXT). Then click **Rebuild knowledge base**.")
    uploaded = st.file_uploader(
        "Upload files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    col_a, col_b = st.columns([1, 1])

    with col_a:
        if st.button("⬆️ Save uploaded files", use_container_width=True):
            if not uploaded:
                st.warning("No files selected.")
            else:
                saved = save_uploaded_files(uploaded)
                if saved:
                    st.success(f"Saved: {', '.join(saved)}")
                else:
                    st.warning("No valid PDF/TXT files were saved.")

    with col_b:
        if st.button("🔄 Rebuild knowledge base", use_container_width=True):
            with st.spinner("Building knowledge base..."):
                delete_index()
                vs = build_vectorstore()
            if vs:
                st.success("Knowledge base rebuilt ✅")
            else:
                st.warning("No documents found in /docs. Upload PDFs/TXT first.")

    st.caption("Tip: If you replace docs, rebuild the knowledge base so answers update.")


# -----------------------------
# Chat UI
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_q = st.chat_input("Ask an HR question…")

if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    vectorstore = load_vectorstore_if_exists()
    if not vectorstore:
        answer = (
            "I don’t have a knowledge base yet.\n\n"
            "Open **Knowledge base**, upload HR documents (PDF/TXT), then click **Rebuild knowledge base**."
        )
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.stop()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieved = retriever.get_relevant_documents(user_q)
    context = "\n\n---\n\n".join([d.page_content for d in retrieved])

    llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.2)

    prompt = f"""
You are Humanio AI, an internal HR assistant for employees.
Answer ONLY using the provided context. If the context does not contain the answer, say you don't know and suggest who to contact or which policy to check.

CONTEXT:
{context}

QUESTION:
{user_q}

ANSWER:
"""

    resp = llm.invoke(prompt)
    answer = resp.content

    with st.chat_message("assistant"):
        st.markdown(answer)

        with st.expander("Sources used"):
            for i, d in enumerate(retrieved, start=1):
                src = d.metadata.get("source", "unknown")
                st.write(f"{i}. {src}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
