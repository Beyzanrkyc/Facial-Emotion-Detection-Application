import streamlit as st

def initialize_session_state():
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if 'guest_mode' not in st.session_state:
        st.session_state['guest_mode'] = False
    if 'page' not in st.session_state:
        st.session_state['page'] = 'login'

def reset_session():
    st.session_state['authenticated'] = False
    st.session_state['guest_mode'] = False
    st.session_state['page'] = 'login'

    if 'email' in st.session_state:
        del st.session_state['email']
    if 'password' in st.session_state:
        del st.session_state['password']