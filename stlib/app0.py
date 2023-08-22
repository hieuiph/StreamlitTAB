import streamlit as st
from PIL import Image

title = """Choose Projects here..."""
description = """ """
def run():
    # Centered Image
    for  i in range(10):
        st.write('')
    
    image = Image.open('stlib/files/wolf3.png') 
    co = st.columns(5)
    with co[2]:
        st.image(image)
    
    titolo = 'Data Scientist'
    st.markdown("<h1 style='text-align: center; padding: 1px; height: 200px;'>"+str(titolo)+"</h1>", unsafe_allow_html=True)

        

if __name__ == "__main__":
     run()