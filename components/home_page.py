import streamlit as st
import pandas as pd
import numpy as np
import random

def get_emotion_insight():
    """Return a random insight about emotions"""
    insights = [
        "Did you know? The ability to recognize emotions is linked to higher empathy and better social relationships.",
        "Research shows that simply naming your emotions can reduce their intensity and help you process them.",
        "There are 7 universally recognized facial expressions: happiness, sadness, anger, fear, disgust, surprise, and contempt.",
        "Your facial expressions can influence how you feel. Forcing a smile can actually improve your mood!",
        "Emotions typically last between 90 seconds and 4 minutes. Anything longer is usually due to us feeding the emotion.",
        "Emotional awareness is a key component of emotional intelligence, which is linked to career success and life satisfaction.",
        "Even micro-expressions that last for a fraction of a second can reveal your true feelings."
    ]
    return random.choice(insights)

def get_emotion_quote():
    """Return a motivational quote about emotions"""
    quotes = [
        "\"The best and most beautiful things in the world cannot be seen or even touched. They must be felt with the heart.\" – Helen Keller",
        "\"I don't want to be at the mercy of my emotions. I want to use them, to enjoy them, and to dominate them.\" – Oscar Wilde",
        "\"No one cares how much you know, until they know how much you care.\" – Theodore Roosevelt",
        "\"Your emotions are the slaves to your thoughts, and you are the slave to your emotions.\" – Elizabeth Gilbert",
        "\"The emotion that can break your heart is sometimes the very one that heals it.\" – Nicholas Sparks",
        "\"Emotion, which is suffering, ceases to be suffering as soon as we form a clear and precise picture of it.\" – Baruch Spinoza",
        "\"There is no feeling, except the extremes of fear and grief, that does not find relief in music.\" – George Eliot"
    ]
    return random.choice(quotes)

def home_content():
    st.title("Welcome to Emotion Detection App")

    if 'authenticated' in st.session_state and st.session_state['authenticated'] and not st.session_state.get('guest_mode', False):
        if 'username' in st.session_state:
            st.write(f"Hello, {st.session_state['username']}! Discover your emotional patterns and enhance your emotional awareness today.")
    else:
        st.write("Discover your emotional patterns and enhance your emotional awareness with our real-time emotion detection technology.")

    with st.container(border=True):
        st.markdown(f"### 💭 *{get_emotion_quote()}*")

    st.write("""
    Understanding our emotions is the first step toward emotional intelligence. Our app helps you:
    - **Recognize** your emotional expressions in real-time
    - **Track** patterns in your emotional states
    - **Gain insights** into your emotional responses
    - **Improve** your emotional awareness
    """)

    st.header("Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Real-time Emotion Detection")
        st.image("https://cdn-icons-png.flaticon.com/512/4370/4370876.png", width=150)
        st.write("""
        See your emotions as they happen:
        - Instant facial analysis
        - Seven emotion categories
        - Live emotion tracking
        """)
    
    with col2:
        st.subheader("Emotion Analytics")
        st.image("https://cdn-icons-png.flaticon.com/512/6588/6588143.png", width=150)
        st.write("""
        Understand your emotional patterns:
        - Detailed visual breakdowns
        - Compare emotions over time
        - Identify dominant emotions
        """)
    
    with col3:
        st.subheader("Personalized Insights")
        st.image("https://cdn-icons-png.flaticon.com/512/2784/2784459.png", width=150)
        st.write("""
        Your emotional data, your insights:
        - Private to your account
        - Long-term pattern recognition
        - Export data for personal use
        """)

    with st.container(border=True):
        st.subheader("💡 Did You Know?")
        st.info(get_emotion_insight())

    st.header("Why Emotional Awareness Matters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Benefits of Emotional Intelligence
        - Better decision making
        - Improved mental health
        - Enhanced relationships
        - Increased resilience to stress
        - More effective communication
        - Higher workplace performance
        """)
    
    with col2:
        st.subheader("Sample Emotion Distribution")
        data = pd.DataFrame({
            'Emotion': ['Happiness', 'Sadness', 'Anger', 'Neutral', 'Surprise', 'Fear', 'Disgust'],
            'Percentage': [42, 18, 12, 35, 22, 8, 5]
        })

    st.header("Psychological Benefits")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("🧠 Cognitive Enhancement"):
            st.write("""
            When you become more aware of your emotions, you engage different parts of your brain, 
            particularly the prefrontal cortex (responsible for reasoning) and the amygdala (the emotional center). 
            Regular emotional awareness exercises strengthen the connections between these areas, 
            improving your ability to process complex emotional situations.
            """)
        
        with st.expander("🛌 Better Stress Management"):
            st.write("""
            Recognizing emotions as they arise gives you more control over your responses.
            This awareness creates a buffer between stimulus and reaction, allowing you to choose
            healthier coping mechanisms and reduce the physical toll of stress on your body.
            """)
    
    with col2:
        with st.expander("💞 Relationship Enhancement"):
            st.write("""
            Understanding your own emotions is the foundation for empathy - the ability to understand
            and share the feelings of others. As you become more emotionally aware, you'll likely
            find improvements in your personal and professional relationships through better communication
            and deeper connections.
            """)
        
        with st.expander("🌟 Personal Growth"):
            st.write("""
            Emotional awareness helps you identify patterns in your reactions, illuminating areas for personal
            growth. By tracking your emotions over time, you can discover triggers and develop strategies
            to transform challenging emotional responses into opportunities for development.
            """)

    st.header("How It Works")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("""
        1. **Facial Detection** - Our app uses advanced computer vision to locate and track faces in your webcam feed
        2. **Feature Extraction** - Key facial landmarks are identified and analyzed for muscle movements and positioning
        3. **Emotion Classification** - Our deep learning model, trained on thousands of facial expressions, classifies your emotion
        4. **Real-time Feedback** - Results are displayed instantly, allowing you to see how your expressions are perceived
        5. **Data Analysis** - Your emotion data is securely stored and analyzed to reveal patterns over time
        6. **Personalized Insights** - Gain valuable insights about your emotional tendencies and patterns
        """)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/3039/3039788.png", width=200)

    st.header("Ready to Start?")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Try Emotion Detection"):
            st.session_state['nav_destination'] = 'Emotion Detection'
            st.rerun()
    
    with col2:
        if st.button("View Analytics"):
            st.session_state['nav_destination'] = 'Analytics'
            st.rerun()

    st.header("Today's Emotional Practice")
    practices = [
        {
            "title": "The 90-Second Rule",
            "description": "The next time you feel a strong emotion, set a timer for 90 seconds. Notice how the physical sensation of the emotion (increased heart rate, tension, etc.) naturally subsides within this time if you don't mentally feed it."
        },
        {
            "title": "Emotion Naming",
            "description": "When you feel an emotion today, try to name it with precision. Instead of just 'angry,' are you frustrated, irritated, or indignant? Specific naming helps your brain process emotions more effectively."
        },
        {
            "title": "Facial Feedback",
            "description": "Try holding a pencil horizontally between your teeth (without letting your lips touch it) for 2 minutes. This forces a smile muscle configuration that can actually improve your mood, demonstrating how facial expressions influence emotions."
        },
        {
            "title": "Emotion Journaling",
            "description": "Take 5 minutes to write down your current emotions and their possible triggers. Don't judge or analyze—just observe and record. This simple practice builds emotional awareness over time."
        },
        {
            "title": "Micro-Expression Spotting",
            "description": "While watching TV today, try to catch the micro-expressions (very brief facial movements) of characters during emotional scenes. This trains your brain to be more attentive to subtle emotional cues."
        }
    ]
    
    daily_practice = random.choice(practices)
    
    with st.container(border=True):
        st.subheader(f"🌱 {daily_practice['title']}")
        st.write(daily_practice['description'])

    st.subheader("Additional Resources")
    st.markdown("""
    - [American Psychological Association - Emotions](https://www.apa.org/topics/emotions)
    - [Yale Center for Emotional Intelligence](http://ei.yale.edu/)
    - [6 Science-Based Benefits of Emotional Intelligence](https://www.healthline.com/health/emotional-intelligence)
    - [Emotion Researchers Network](https://emotionresearchers.com/)
    """)

if __name__ == "__main__":
    home_content()