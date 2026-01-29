import os
import streamlit as st

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Humanio AI", page_icon="🤝", layout="wide")
st.title("Humanio AI — HR Assistant")

DOCS_PATH = "docs"
INDEX_PATH = "data/faiss_index"

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found in Streamlit Secrets.")
    st.stop()


# -------- Load documents --------
def load_documents():
    docs = []
    if not os.path.exists(DOCS_PATH):
        os.makedirs(DOCS_PATH, exist_ok=True)

    for root, _, files in os.walk(DOCS_PATH):
        for f in files:
            fp = os.path.join(root, f)
            if f.lower().endswith(".pdf"):
                docs.extend(PyPDFLoader(fp).load())
            elif f.lower().endswith(".txt"):
                docs.extend(TextLoader(fp, encoding="utf-8").load())
    return docs


# -------- Vector store --------
def get_vectorstore():
    embeddings = OpenAIEmbeddings(api_key=api_key)

    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    docs = load_documents()
    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    vs = FAISS.from_documents(chunks, embeddings)
    os.makedirs(INDEX_PATH, exist_ok=True)
    vs.save_local(INDEX_PATH)
    return vs


def delete_index():
    if os.path.exists(INDEX_PATH):
        for root, dirs, files in os.walk(INDEX_PATH, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(INDEX_PATH)


# -------- Sidebar --------
st.sidebar.header("Knowledge Base")
st.sidebar.write("Upload HR PDFs/TXT into the `/docs` folder in GitHub.")

if st.sidebar.button("Rebuild index"):
    with st.spinner("Rebuilding knowledge base..."):
        delete_index()
        vectorstore = get_vectorstore()

    if vectorstore:
        st.sidebar.success("Index rebuilt ✅")
    else:
        st.sidebar.warning("No documents found in /docs.")


# -------- Chat --------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_q = st.chat_input("Ask an HR question…")

if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})

    with st.chat_message("user"):
        st.markdown(user_q)

    vectorstore = get_vectorstore()

    if not vectorstore:
        answer = "No HR documents found. Upload files to `/docs` and click 'Rebuild index'."
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.stop()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieved = retriever.get_relevant_documents(user_q)

    context = "\n\n---\n\n".join([d.page_content for d in retrieved])

    llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.2)

    prompt = f"""
You are Humanio AI, an internal HR assistant.
Answer ONLY using the provided context.
If unsure, say you don't know and suggest checking HR.

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

