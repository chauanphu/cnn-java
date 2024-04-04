import streamlit as st
from PIL import Image

st.title("Image Upload")
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)