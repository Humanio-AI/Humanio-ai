import os
import time
import shutil
import streamlit as st
from PIL import Image

from openai import RateLimitError  # from openai python SDK

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

if not os.path.exists(LOGO_PATH):
    st.set_page_config(page_title="Humanio AI", page_icon="🤝", layout="wide")
    st.error("Upload logo.png to repo root.")
    st.stop()

logo = Image.open(LOGO_PATH)
st.set_page_config(page_title="Humanio AI", page_icon=logo, layout="wide")

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY missing in Streamlit Secrets.")
    st.stop()

# -----------------------------
# Session
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

def clear_chat():
    st.session_state.messages = []

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

def load_vectorstore_if_exists():
    embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
    if os.path.exists(INDEX_PATH):
        return
