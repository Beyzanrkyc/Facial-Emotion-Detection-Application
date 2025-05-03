import streamlit as st
import re
from utils.db_handler import save_user, verify_duplicate_user
import time

def is_valid_email(email):
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None

def signup_page(extra_input_params=False, confirmPass=False):
    if st.button("Back to Login"):
        st.session_state['page'] = 'login'
        st.rerun()
    
    with st.empty().container(border=True):
        st.title("Sign Up Page")

        st.session_state['email'] = st.text_input("Email")
        if st.session_state['email'] and not is_valid_email(st.session_state['email']):
            st.error("Please enter a valid email address")

        st.session_state['password'] = st.text_input("Password", type='password')
        confirm_password = st.text_input("Confirm Password", type='password')
        st.session_state['username'] = st.text_input("username")
        st.session_state['age'] = st.number_input("age", min_value=0, step=1)

        if (st.session_state['email'] and st.session_state['password'] and 
            st.session_state['username'] and st.session_state['age'] >= 0 and
            st.session_state['password'] == confirm_password):
            
            if st.button("Register"):
                if verify_duplicate_user(st.session_state['email']):
                    st.error("User already exists")
                else:
                    user_data = {
                        'username': st.session_state['username'],
                        'age': st.session_state['age']
                    }

                    success, user_id = save_user(st.session_state['email'], st.session_state['password'], user_data)
                    
                    if success:
                        st.success("Registration successful! Redirecting to login...")
                        time.sleep(2)
                        st.session_state['page'] = 'login'
                        st.rerun()
                    else:
                        st.error(f"Registration failed: {user_id}") 
        else:
            if st.session_state['password'] != confirm_password and st.session_state['password']:
                st.error("Passwords do not match")
            elif st.button("Register"):
                st.error("Please fill in all required fields")