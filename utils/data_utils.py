import os
import pandas as pd
import numpy as np
from datetime import datetime
import psycopg2
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

db_params = {
    "dbname":  os.getenv("DB_NAME", "emotion_detection_app"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "postgres"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

def get_db_connection():
    """Create and return a database connection"""
    try:
        conn = psycopg2.connect(**db_params)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def save_emotion_data(emotion_counts, user_id=None, session_id=None):
    """
    Save emotion detection data to the database (or CSV fallback)
    Only saves data for authenticated users, not guests
    
    Args:
        emotion_counts (dict): Dictionary with emotion counts
        user_id (str, optional): User identifier for logged-in users. None for guest users.
        session_id (str, optional): Session identifier. Defaults to timestamp.
    
    Returns:
        str: The session ID used for saving the data, or None if in guest mode
    """
    # If using guest mode, don't save data
    if user_id == "guest" or ('guest_mode' in st.session_state and st.session_state['guest_mode']):
        print("Guest mode active - no data saved")
        return session_id
    
    # Try saving to database first
    if save_to_db(emotion_counts, user_id, session_id):
        return session_id
    
    # Fallback to CSV storage
    return save_to_csv(emotion_counts, user_id, session_id)

def save_to_db(emotion_counts, user_id=None, session_id=None):
    """Save emotion data to database"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        # Don't save anything for guest mode
        if user_id == "guest":
            return False
            
        # Generate session ID if not provided
        if session_id is None:
            session_id = f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        cur = conn.cursor()
        
        # Check if emotion_sessions table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'emotion_sessions'
            )
        """)
        
        if not cur.fetchone()[0]:
            # Create emotion_sessions table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS emotion_sessions (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255),
                    session_name VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
        # Check if emotion_data table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'emotion_data'
            )
        """)
        
        if not cur.fetchone()[0]:
            # Create emotion_data table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS emotion_data (
                    id SERIAL PRIMARY KEY,
                    session_id INTEGER REFERENCES emotion_sessions(id) ON DELETE CASCADE,
                    emotion VARCHAR(50) NOT NULL,
                    count INTEGER NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
        # Check if confidence_score column exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'emotion_data' AND column_name = 'confidence_score'
            )
        """)
        
        has_confidence = cur.fetchone()[0]
        
        if not has_confidence:
            # Add confidence_score column if it doesn't exist
            cur.execute("""
                ALTER TABLE emotion_data 
                ADD COLUMN confidence_score FLOAT
            """)
            print("Added confidence_score column to emotion_data table")
        
        # Check if is_guest_session column exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'emotion_sessions' AND column_name = 'is_guest_session'
            )
        """)
        
        has_guest_column = cur.fetchone()[0]
        
        # Insert session into database based on column availability
        if has_guest_column:
            cur.execute(
                "INSERT INTO emotion_sessions (user_id, session_name, is_guest_session) VALUES (%s, %s, %s) RETURNING id",
                (str(user_id), session_id, False)
            )
        else:
            cur.execute(
                "INSERT INTO emotion_sessions (user_id, session_name) VALUES (%s, %s) RETURNING id",
                (str(user_id), session_id)
            )
        
        db_session_id = cur.fetchone()[0]
        
        # Get confidence scores for emotions (if available in session state)
        confidence_scores = {}
        if 'emotion_confidence' in st.session_state:
            confidence_scores = st.session_state.emotion_confidence
        
        # Insert emotion data
        for emotion, count in emotion_counts.items():
            # Get confidence score if available
            conf_score = confidence_scores.get(emotion, None)
            
            cur.execute(
                "INSERT INTO emotion_data (session_id, emotion, count, confidence_score) VALUES (%s, %s, %s, %s)",
                (db_session_id, emotion, count, conf_score)
            )
        
        conn.commit()
        cur.close()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Database error saving emotion data: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return False

def save_to_csv(emotion_counts, user_id=None, session_id=None):
    """Save emotion data to CSV file (fallback method)"""
    # Don't save anything for guest mode
    if user_id == "guest":
        return session_id
        
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate session ID if not provided
    if session_id is None:
        session_id = f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create timestamp
    timestamp = datetime.now()
    
    # Get confidence scores for emotions (if available)
    confidence_scores = {}
    if 'emotion_confidence' in st.session_state:
        confidence_scores = st.session_state.emotion_confidence
    
    # Create a dataframe from the emotion counts
    data = []
    for emotion, count in emotion_counts.items():
        row_data = {
            'user_id': str(user_id) if user_id is not None else "guest",
            'session_id': session_id,
            'timestamp': timestamp,
            'emotion': emotion,
            'count': count
        }
        
        # Add confidence score if available
        if emotion in confidence_scores:
            row_data['confidence_score'] = confidence_scores.get(emotion)
            
        data.append(row_data)
    
    df = pd.DataFrame(data)
    
    # Save to CSV - append if file exists, create if it doesn't
    if os.path.exists('data/emotion_data.csv'):
        try:
            # Check if existing CSV has confidence_score column
            existing_df = pd.read_csv('data/emotion_data.csv')
            if 'confidence_score' not in existing_df.columns and 'confidence_score' in df.columns:
                # Add confidence_score column to existing data with NaN values
                existing_df['confidence_score'] = np.nan
                # Save the updated file
                existing_df.to_csv('data/emotion_data.csv', index=False)
            
            # Now append the new data
            df.to_csv('data/emotion_data.csv', mode='a', header=False, index=False)
        except Exception as e:
            print(f"Error updating existing CSV: {e}")
            # Create a new CSV as fallback
            df.to_csv('data/emotion_data.csv', index=False)
    else:
        df.to_csv('data/emotion_data.csv', index=False)
    
    return session_id

def load_emotion_data(user_id=None):
    """
    Load saved emotion data for specific user (from database or CSV fallback)
    
    Args:
        user_id (str, optional): User identifier to filter data. If None, all data is returned.
                                 For guest users, no data is returned.
    
    Returns:
        DataFrame: Pandas DataFrame with emotion data or None if no data found
    """
    # Don't return any data for guest mode
    if user_id == "guest" or ('guest_mode' in st.session_state and st.session_state['guest_mode']):
        return None
    
    # Try loading from database first
    df = load_from_db(user_id)
    if df is not None and not df.empty:
        return df
    
    # Fallback to CSV storage
    return load_from_csv(user_id)

def load_from_db(user_id=None):
    """Load emotion data from database"""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        cur = conn.cursor()
        
        # Check if tables exist
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'emotion_data'
            )
        """)
        
        if not cur.fetchone()[0]:
            return None
        
        # Check if confidence_score column exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'emotion_data' AND column_name = 'confidence_score'
            )
        """)
        has_confidence = cur.fetchone()[0]
        
        # Build query based on user_id and available columns
        if has_confidence:
            select_fields = """
                d.emotion, 
                d.count, 
                d.timestamp, 
                d.confidence_score,
                s.session_name as session_id, 
                s.user_id
            """
        else:
            select_fields = """
                d.emotion, 
                d.count, 
                d.timestamp,
                s.session_name as session_id, 
                s.user_id
            """
            
        query = f"""
            SELECT 
                {select_fields}
            FROM 
                emotion_data d
            JOIN 
                emotion_sessions s ON d.session_id = s.id
        """
        
        params = []
        if user_id is not None:
            query += " WHERE s.user_id = %s"
            params.append(str(user_id))  # Convert user_id to string
        
        # Execute query
        cur.execute(query, params)
        
        # Convert to dataframe
        if has_confidence:
            columns = ['emotion', 'count', 'timestamp', 'confidence_score', 'session_id', 'user_id']
        else:
            columns = ['emotion', 'count', 'timestamp', 'session_id', 'user_id']
            
        data = cur.fetchall()
        
        cur.close()
        conn.close()
        
        if not data:
            return None
            
        df = pd.DataFrame(data, columns=columns)
        return df
        
    except Exception as e:
        print(f"Database error loading emotion data: {e}")
        if conn:
            conn.close()
        return None

def load_from_csv(user_id=None):
    """Load emotion data from CSV file (fallback method)"""
    if os.path.exists('data/emotion_data.csv'):
        try:
            # Use on_bad_lines='skip' to skip problematic rows
            df = pd.read_csv('data/emotion_data.csv', on_bad_lines='skip')
            # Convert timestamp strings to datetime objects
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by user_id if provided and if column exists
            if user_id is not None and 'user_id' in df.columns:
                df = df[df['user_id'] == str(user_id)]  # Convert user_id to string
                
            return df
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            # Try with a different approach as a last resort
            try:
                df = pd.read_csv('data/emotion_data.csv', engine='python')
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                if user_id is not None and 'user_id' in df.columns:
                    df = df[df['user_id'] == str(user_id)]  # Convert user_id to string
                return df
            except Exception as e2:
                print(f"Second attempt to load CSV failed: {e2}")
                return None
    else:
        return None

def get_user_session_stats(user_id=None):
    """
    Get statistics about user sessions
    
    Args:
        user_id (int, optional): User ID to filter by
    
    Returns:
        dict: Dictionary with session statistics
    """
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        cur = conn.cursor()
        
        query = """
            SELECT 
                COUNT(DISTINCT s.id) as session_count,
                MIN(s.created_at) as first_session,
                MAX(s.created_at) as last_session,
                COUNT(d.id) as total_records
            FROM 
                emotion_sessions s
            LEFT JOIN 
                emotion_data d ON s.id = d.session_id
        """
        
        params = []
        if user_id:
            query += " WHERE s.user_id = %s"
            params.append(str(user_id))  # Convert user_id to string
        
        cur.execute(query, params)
        result = cur.fetchone()
        
        if result:
            stats = {
                'session_count': result[0],
                'first_session': result[1],
                'last_session': result[2],
                'total_records': result[3]
            }
            return stats
        
        return None
    except Exception as e:
        print(f"Error getting session stats: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_emotion_categories():
    """
    Get all emotion categories from database
    
    Returns:
        DataFrame: Pandas DataFrame with emotion categories
    """
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        cur = conn.cursor()
        
        # Check if emotion_categories table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'emotion_categories'
            )
        """)
        
        if not cur.fetchone()[0]:
            # If table doesn't exist, return default categories
            default_categories = pd.DataFrame({
                'id': range(7),
                'name': ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear'],
                'description': ['No strong emotion', 'Joy or pleasure', 'Unexpected reaction', 
                               'Feeling of loss', 'Strong displeasure', 'Revulsion', 'Response to threat'],
                'display_color': ['#9E9E9E', '#4CAF50', '#FF9800', '#2196F3', '#F44336', '#795548', '#673AB7']
            })
            return default_categories
        
        cur.execute("SELECT id, name, description, display_color FROM emotion_categories")
        data = cur.fetchall()
        
        if data:
            df = pd.DataFrame(data, columns=['id', 'name', 'description', 'display_color'])
            return df
        else:
            # If no data, return default categories
            default_categories = pd.DataFrame({
                'id': range(7),
                'name': ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear'],
                'description': ['No strong emotion', 'Joy or pleasure', 'Unexpected reaction', 
                               'Feeling of loss', 'Strong displeasure', 'Revulsion', 'Response to threat'],
                'display_color': ['#9E9E9E', '#4CAF50', '#FF9800', '#2196F3', '#F44336', '#795548', '#673AB7']
            })
            return default_categories
    except Exception as e:
        print(f"Error getting emotion categories: {e}")
        return None
    finally:
        if conn:
            conn.close()