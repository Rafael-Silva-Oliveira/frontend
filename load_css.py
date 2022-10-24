import streamlit as st
import sys
sys.path.insert(0, 'frontend\pages')

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)