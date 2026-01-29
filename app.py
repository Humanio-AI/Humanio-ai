import os
import shutil
import streamlit as st
from PIL import Image

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# Config (logo, docs, index)
# -----------------------------
LOGO_PATH = "logo.png"           # put logo.png in repo root
DOCS_PATH = "docs"               # upload docs here via GitHub
INDEX_PATH = "data/faiss_index"  # cache folder on Streamlit

# Page icon MUST be logo file
if not os.path.exists(LOGO_PATH):
    st.set_page_config(page_title="Humanio AI", page_icon="🤝", layout="wide")
    st.error("logo.png not found in repo root. Upload your logo as 'logo.png'.")
    st.stop()

logo_img = Image.open(LOGO_PATH)
st.set_page_config(page_title="Humanio AI", page_icon=logo_img, layout="wide")

# -----------------------------
# Secrets
# -----------------------------
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found in Streamlit Secrets.")
    st.stop()

# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

def clear_chat():
    st.session_state.messages = []

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

def get_vectorstore():
    vs = load_vectorstore_if_exists()
    if vs:
        return vs
    # First run: build automatically
    delete_index()
    return build_vectorstore()

# -----------------------------
# UI (Centered logo + title + centered New chat button)
# -----------------------------
st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns([3, 2, 3])
with c2:
    st.image(logo_img, width=300)

st.markdown(
    """
    <div style="text-align:center; margin-top: 6px;">
        <h1 style="margin-bottom: 0.2rem;">Humanio AI</h1>
        <div style="opacity:0.78; font-size: 1.05rem;">HR Team</div>
    </div>
    """,
    unsafe_allow_html=True,
)

c4, c5, c6 = st.columns([3, 2, 3])
with c5:
    if st.button("New chat", use_container_width=True):
        clear_chat()
        st.rerun()

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
st.divider()

# -----------------------------
# Chat history
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# Chat input + response
# -----------------------------
user_q = st.chat_input("Input HR Query"\)

if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    vectorstore = get_vectorstore()

    # No docs / KB not built
    if not vectorstore:
        answer = (
            "I don’t have any HR documents yet.\n\n"
            "Upload HR PDFs/TXT files into the **/docs** folder in GitHub, then refresh this page."
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
""".strip()

    resp = llm.invoke(prompt)
    answer = resp.content

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
