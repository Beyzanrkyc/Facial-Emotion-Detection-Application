import streamlit as st
from utils.db_handler import authenticate_user
import time

def login_page(guest_mode=False):
    with st.empty().container(border=True):
        col1, _, col2 = st.columns([10,1,10])
        
        with col1:
            st.write("")
            st.write("")
            # st.video("data/demo.mp4", autoplay=True, loop=True, muted=True)
            st.image("data/demo.png")
        
        with col2:
            st.title("Login Page")

            email = st.text_input("E-mail")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                time.sleep(2)
                if not (email and password):
                    st.error("Please provide email and password")
                else:
                    success, user_data = authenticate_user(email, password)
                    if success:
                        st.session_state['authenticated'] = True
                        st.session_state['page'] = 'app'
                        st.session_state['email'] = email

                        if user_data:
                            st.session_state['user_id'] = user_data.get('user_id')
                            st.session_state['username'] = user_data.get('username')
                            st.session_state['age'] = user_data.get('age')
                        
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid login credentials")

            if st.button("Sign Up"):
                st.session_state['page'] = 'signup'
                st.rerun()
                
            if guest_mode:
                if st.button("Continue as Guest"):
                    st.session_state['guest_mode'] = True
                    st.session_state['authenticated'] = True
                    st.session_state['page'] = 'app'
                    st.session_state['user_id'] = "guest"
                    st.session_state['username'] = "Guest User"
                    st.rerun()