import psycopg2
import bcrypt
import os
from dotenv import load_dotenv

load_dotenv()

db_params = {
    "dbname":  os.getenv("DB_NAME", "emotion_detection_app"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "postgres"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

def get_db_connection():
    try:
        conn = psycopg2.connect(**db_params)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def authenticate_user(email, password):

    conn = get_db_connection()
    if not conn:
        return False, None
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, email, hash_password, username, age FROM users WHERE email = %s", (email,))
        user = cur.fetchone()

        if user:
            if bcrypt.checkpw(password.encode('utf-8'), user[2].encode('utf-8')):
                user_data = {
                    'user_id': user[0],
                    'email': user[1],
                    'username': user[3],
                    'age': user[4]
                }
                return True, user_data
        
        return False, None
    except Exception as e:
        print(f"Authentication error: {e}")
        return False, None
    finally:
        if conn:
            conn.close()

def verify_duplicate_user(email):

    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT EXISTS(SELECT 1 FROM users WHERE email = %s)", (email,))
        exists = cur.fetchone()[0]
        return exists
    except Exception as e:
        print(f"Error checking for duplicate user: {e}")
        return False
    finally:
        if conn:
            conn.close()

def save_user(email, password, user_data):

    conn = get_db_connection()
    if not conn:
        return False, "Database connection failed"
    
    try:
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        username = user_data.get('username', email.split('@')[0])
        age = user_data.get('age', 0)
        
        cur = conn.cursor()
 
        cur.execute(
            "INSERT INTO users (email, hash_password, username, age) VALUES (%s, %s, %s, %s) RETURNING id",
            (email, hashed_password.decode('utf-8'), username, age)
        )

        user_id = cur.fetchone()[0]
        conn.commit()
        
        return True, user_id
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        return False, "Email or username already exists"
    except Exception as e:
        if conn:
            conn.rollback()
        return False, f"Error saving user: {e}"
    finally:
        if conn:
            conn.close()

def get_user_by_id(user_id):
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, email, username, age FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
        
        if user:
            return {
                'user_id': user[0],
                'email': user[1],
                'username': user[2],
                'age': user[3]
            }
        return None
    except Exception as e:
        print(f"Error getting user: {e}")
        return None
    finally:
        if conn:
            conn.close()