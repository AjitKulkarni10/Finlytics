import streamlit as st
import os
import requests


def does_file_have_pdf_extension(file):
    if not (file.name.endswith(".pdf")):
        st.warning("Not a valid PDF file.", icon="⚠️")
        return False
    return True



def load_lottieurl(
    url="https://lottie.host/4465d9cf-dd0a-4b6a-9f75-16cbcfa2ddbb/uJocIZNlqN.json",
):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def store_pdf_file(file, save_dir):
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path
