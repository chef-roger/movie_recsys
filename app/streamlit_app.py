"""
JOINT FILE — built together in Phase 7-8, once recommend() in interfaces.py
is fully wired up.

Usage: streamlit run app/streamlit_app.py
"""
import streamlit as st
# from interfaces import recommend

st.title("Movie Recommender")

user_id = st.number_input("User ID", min_value=1, step=1, value=1)
mode = st.radio("Mode", ["ranked", "explore"])

if st.button("Get recommendations"):
    st.write("TODO: call recommend(user_id, mode) once it's implemented")
    # movie_ids = recommend(user_id, mode=mode, k=10)
    # for mid in movie_ids:
    #     st.write(mid)  # replace with title lookup + poster
