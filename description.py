import streamlit as st

st.title("実験の手順")
st.markdown("""
これは対話セッションのみを行うアプリです。
本来の実験手順とは異なります。
""")

if st.button("対話セッションを始める"):
    st.session_state.current_page = "dialogue"
    st.rerun()
