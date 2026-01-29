import streamlit as st

st.title("Humanio AI - Secrets Test")

key = st.secrets.get("OPENAI_API_KEY")

if not key:
    st.error("OPENAI_API_KEY not found in Streamlit Secrets.")
else:
    st.success("OPENAI_API_KEY loaded from Streamlit Secrets ✅")
    st.write("Key length:", len(key))
