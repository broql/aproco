import streamlit as st
from PIL import Image

from dotenv import load_dotenv

from _chat import run_chat
from _landing import run_landing_page


load_dotenv()

columns = st.columns(5)
columns[1].text('')
columns[1].text('')
columns[1].image(Image.open('gui/images/aproco.png'), width=150)
columns[3].image(Image.open('gui/images/nd.png'), width=80)

if 'email' not in st.session_state:
    run_landing_page(run_chat)
else:
    run_chat()