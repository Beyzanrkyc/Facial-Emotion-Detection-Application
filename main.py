import streamlit as st
import os
import sys
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from authentication.login_page import login_page
from authentication.signup_page import signup_page

from components.home_page import home_content
from components.analytics_page import analytics_page
from components.emotion_detection_page import emotion_detection_page

from utils.session_utils import initialize_session_state, reset_session

st.set_page_config(
    page_title="Emotion Detection App",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def main():
    initialize_session_state()
    if st.session_state['page'] == 'login':
        login_page(guest_mode=True)
    elif st.session_state['page'] == 'signup':
        signup_page(extra_input_params=True, confirmPass=True)
    elif st.session_state['authenticated'] or st.session_state['guest_mode']:
        with st.sidebar:
            st.title("Navigation")
            if st.session_state['guest_mode']:
                st.info("Guest Mode")
            else:
                st.success("Logged In")
                if 'username' in st.session_state:
                    st.write(f"Welcome, {st.session_state['username']}")

            if st.button("Logout"):
                reset_session()
                st.rerun()

            st.subheader("Pages")
            page_selection = st.radio(
                "Select a page:",
                ["Home", "Emotion Detection", "Analytics"],
                label_visibility="collapsed"
            )

        if page_selection == "Home":
            home_content()
        elif page_selection == "Emotion Detection":
            emotion_detection_page()
        elif page_selection == "Analytics":
            analytics_page()
    else:
        st.session_state['page'] = 'login'
        st.rerun()

if __name__ == "__main__":
    main()