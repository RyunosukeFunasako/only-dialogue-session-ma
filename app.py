import streamlit as st

description_page = st.Page("description.py", title="実験の説明", icon="✅")
dialogue_page = st.Page("dialogue-session/dialogue.py", title="対話セッション", icon="👩‍⚕️")

if "current_page" not in st.session_state:
    st.session_state.current_page = "description"

# 対話履歴記録
if "dialogue_history" not in st.session_state:
    st.session_state.dialogue_history = []

# 逸脱履歴記録
if "deviation_history" not in st.session_state:
    st.session_state.deviation_history = []

if st.session_state.current_page == "description":
    pg = st.navigation([description_page])
elif st.session_state.current_page == "dialogue":
    pg = st.navigation([dialogue_page])

pg.run()