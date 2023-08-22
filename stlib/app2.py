import streamlit as st
description = "My second page"
title = 'TITLE 2' 
def run():
   st.header("This is my second page")

   st.markdown("""
<style>
.centered-text {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 50vh; /* Imposta l'altezza al 100% della viewport */
}
</style>
<div class="centered-text">
    <h1>🚧👷🏻‍♂️ Works in progress 👷🏻‍♂️🚧 🏗️</h1>
</div>
""", unsafe_allow_html=True)
   if __name__ == "__main__":
     run()