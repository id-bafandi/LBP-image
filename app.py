import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern

st.set_page_config(page_title="Aplikasi Ekstraksi LBP", layout="wide")
st.title("Aplikasi Local Binary Pattern (LBP)")
st.write("Unggah gambar dari komputer kamu untuk melihat hasil ekstraksi tekstur LBP secara langsung!")

uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Tangani RGB dan RGBA
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = img_array

    radius = 1
    n_points = 8 * radius
    lbp_result = local_binary_pattern(gray_image, n_points, radius, method='uniform')

    # Normalisasi agar gambar LBP terlihat jelas
    lbp_normalized = (lbp_result / lbp_result.max() * 255).astype(np.uint8)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gambar Asli")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Hasil LBP")
        st.image(lbp_normalized, use_container_width=True)


st.write(".")