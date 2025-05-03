import streamlit as st
import pandas as pd
import os
import hashlib
import uuid
from datetime import datetime

def make_hashed_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def create_user_directory(username):

    user_dir = f"data/users/{username}"
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def initialize_user_system():
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/users", exist_ok=True)

    if not os.path.exists("data/users.csv"):
        users_df = pd.DataFrame(columns=["username", "password", "created_at"])
        users_df.to_csv("data/users.csv", index=False)

def register_user(username, password):

    if user_exists(username):
        return False

    hashed_password = make_hashed_password(password)

    create_user_directory(username)

    users_df = pd.read_csv("data/users.csv")
    new_user = pd.DataFrame({
        "username": [username],
        "password": [hashed_password],
        "created_at": [datetime.now()]
    })
    users_df = pd.concat([users_df, new_user], ignore_index=True)
    users_df.to_csv("data/users.csv", index=False)
    
    return True

def authenticate_user(username, password):

    if not user_exists(username):
        return False
    
    users_df = pd.read_csv("data/users.csv")
    user_record = users_df[users_df["username"] == username]
    
    if user_record.empty:
        return False
    
    hashed_password = make_hashed_password(password)
    return user_record.iloc[0]["password"] == hashed_password

def user_exists(username):

    if not os.path.exists("data/users.csv"):
        return False
    
    users_df = pd.read_csv("data/users.csv")
    return username in users_df["username"].values

def get_user_data_path(username, filename="emotion_data.csv"):

    return f"data/users/{username}/{filename}"

def auth_page():

    initialize_user_system()

    if "user_authenticated" in st.session_state and st.session_state["user_authenticated"]:
        return True
    
    st.title("Emotion Tracking - User Authentication")

    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.header("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state["user_authenticated"] = True
                st.session_state["username"] = username
                st.success("Login successful!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")
    
    with tab2:
        st.header("Register")
        new_username = st.text_input("Choose Username", key="reg_username")
        new_password = st.text_input("Choose Password", type="password", key="reg_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        
        if st.button("Register"):
            if new_password != confirm_password:
                st.error("Passwords do not match")
            elif len(new_username) < 3:
                st.error("Username must be at least 3 characters long")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters long")
            else:
                if register_user(new_username, new_password):
                    st.success("Registration successful! You can now login.")
                    st.session_state["reg_username"] = ""
                    st.session_state["reg_password"] = ""
                    st.session_state["confirm_password"] = ""
                else:
                    st.error("Username already exists")

    st.sidebar.info("This application allows you to track and analyze facial emotions in real-time. Sign up to get started!")
    
    return False

def logout_user():
    if "user_authenticated" in st.session_state:
        st.session_state["user_authenticated"] = False
    if "username" in st.session_state:
        del st.session_state["username"]