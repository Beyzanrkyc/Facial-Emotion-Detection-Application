import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import os
load_dotenv()

db_params = {
    "dbname":  os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

def create_database():
    conn = psycopg2.connect(dbname="postgres", **{k: v for k, v in db_params.items() if k != "dbname"})
    conn.autocommit = True
    cur = conn.cursor()
    
    cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_params["dbname"])))
    
    cur.close()
    conn.close()

def create_tables():
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            hash_password TEXT NOT NULL,
            username VARCHAR(100) UNIQUE NOT NULL,
            age INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP WITH TIME ZONE
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS emotion_categories (
            id SERIAL PRIMARY KEY,
            name VARCHAR(50) UNIQUE NOT NULL,
            description TEXT,
            display_color VARCHAR(20)
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS emotion_sessions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            session_name VARCHAR(255) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            device_info TEXT,
            is_guest_session BOOLEAN DEFAULT FALSE
        );
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS emotion_data (
            id SERIAL PRIMARY KEY,
            session_id INTEGER REFERENCES emotion_sessions(id) ON DELETE CASCADE,
            emotion VARCHAR(50) NOT NULL,
            count INTEGER NOT NULL,
            confidence_score FLOAT,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cur.execute("""
        INSERT INTO emotion_categories (name, description, display_color)
        VALUES 
            ('happiness', 'State of well-being and contentment', '#4CAF50'),
            ('sadness', 'Emotional pain associated with disadvantage, loss, grief', '#2196F3'),
            ('anger', 'Strong feeling of annoyance, displeasure, or hostility', '#F44336'),
            ('neutral', 'No strong or obvious emotion', '#9E9E9E'),
            ('surprise', 'Brief mental and physiological state resulting from unexpected event', '#FF9800'),
            ('fear', 'Emotion induced by perceived danger or threat', '#673AB7'),
            ('disgust', 'Aversion or repulsion toward something offensive', '#795548')
        ON CONFLICT (name) DO NOTHING;
    """)
    
    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    try:
        create_database()
        print("Database created successfully.")
    except psycopg2.Error as e:
        print(f"Error creating database: {e}")

    try:
        create_tables()
        print("Tables created successfully.")
    except psycopg2.Error as e:
        print(f"Error creating tables: {e}")