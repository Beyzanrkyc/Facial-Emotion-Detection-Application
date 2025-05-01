import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from utils.data_utils import load_emotion_data
import matplotlib.pyplot as plt

def analytics_page():
    st.title("Emotion Analytics")

    if 'guest_mode' in st.session_state and st.session_state['guest_mode']:
        st.warning("You're in guest mode. No data is saved or available for analytics. Sign in to save and analyze your emotion data.")

        st.subheader("Sample Emotion Distribution (Example Only)")
        sample_data = pd.DataFrame({
            'Emotion': ['Happiness', 'Sadness', 'Anger', 'Neutral', 'Surprise', 'Fear', 'Disgust'],
            'Percentage': [42, 18, 12, 35, 22, 8, 5]
        })
        st.bar_chart(sample_data.set_index('Emotion'))
        
        st.info("Sign in to track your own emotion patterns over time!")
        return
    
    st.write("View detailed analytics of your emotion detection sessions.")

    user_id = None
    if 'authenticated' in st.session_state and st.session_state['authenticated']:
        if 'user_id' in st.session_state:
            user_id = st.session_state['user_id']
            st.success(f"Showing analytics for {st.session_state.get('username', 'your')} account")

    df = load_emotion_data(user_id)
    
    if df is None or df.empty:
        st.info("No saved emotion data found. Use the Emotion Detection page to collect and save data first.")

        st.subheader("Sample Emotion Distribution (Example)")
        sample_data = pd.DataFrame({
            'Emotion': ['Happiness', 'Sadness', 'Anger', 'Neutral', 'Surprise', 'Fear', 'Disgust'],
            'Percentage': [42, 18, 12, 35, 22, 8, 5]
        })
        st.bar_chart(sample_data.set_index('Emotion'))
        
        st.markdown("""
        ### Getting Started
        To collect your own emotion data:
        1. Go to the **Emotion Detection** page
        2. Click **Start Webcam**
        3. Make different facial expressions
        4. Click **Save Data** to store your emotions
        5. Return here to see your emotion analytics
        """)
        return

    has_confidence = 'confidence_score' in df.columns and not df['confidence_score'].isna().all()

    sessions = df['session_id'].unique()
    session_option = st.selectbox(
        "Select session to analyze:",
        ["All Sessions"] + list(sessions)
    )
    
    if session_option != "All Sessions":
        filtered_df = df[df['session_id'] == session_option]
    else:
        filtered_df = df

    st.subheader("Time Range")
    min_date = filtered_df['timestamp'].min().date()
    max_date = filtered_df['timestamp'].max().date()
    
    date_range = st.date_input(
        "Select date range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        end_date = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        date_filtered_df = filtered_df.copy()
        date_filtered_df['timestamp'] = date_filtered_df['timestamp'].dt.tz_localize(None)
        
        date_filtered_df = date_filtered_df[
            (date_filtered_df['timestamp'] >= pd.Timestamp(start_date)) & 
            (date_filtered_df['timestamp'] <= pd.Timestamp(end_date))
        ]
    else:
        date_filtered_df = filtered_df

    if not date_filtered_df.empty:
        with st.container(border=True):
            st.subheader("Data Summary")

            total_emotions = date_filtered_df['count'].sum()
            emotion_totals = date_filtered_df.groupby('emotion')['count'].sum().reset_index()
            top_emotion = emotion_totals.loc[emotion_totals['count'].idxmax()]

            avg_confidence = None
            if has_confidence:
                avg_confidence = date_filtered_df['confidence_score'].mean()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Emotions", f"{total_emotions:,}")
            with col2:
                st.metric("Most Common", f"{top_emotion['emotion'].capitalize()}")
            with col3:
                if avg_confidence is not None:
                    st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                else:
                    st.metric("Sessions", f"{len(date_filtered_df['session_id'].unique())}")
        
        with st.container(border=True):
            st.subheader("Session Summary")
            emotion_totals = date_filtered_df.groupby(['session_id', 'emotion'])['count'].sum().reset_index()
            emotion_pivot = emotion_totals.pivot(index='session_id', columns='emotion', values='count').fillna(0)

            st.dataframe(emotion_pivot.style.highlight_max(axis=1, color='lightgreen'), use_container_width=True)
            
        with st.container(border=True):
            st.subheader("Emotion Distribution")
            
            emotion_totals_only = emotion_totals.groupby('emotion')['count'].sum().reset_index()
            emotion_totals_only = emotion_totals_only.sort_values('count', ascending=False)

            tab1, tab2 = st.tabs(["Bar Chart", "Pie Chart"])
            
            with tab1:
                st.bar_chart(emotion_totals_only.set_index('emotion'))
                
            with tab2:
                fig, ax = plt.subplots(figsize=(10, 8))

                sorted_data = emotion_totals_only.sort_values('count', ascending=False)

                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

                wedges, texts, autotexts = ax.pie(
                    sorted_data['count'], 
                    labels=None, 
                    autopct='%1.1f%%',
                    startangle=90,
                    shadow=False,
                    colors=colors,
                    textprops={'fontsize': 14, 'weight': 'bold'},
                    wedgeprops={'edgecolor': 'w', 'linewidth': 1}
                )

                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(12)
                    autotext.set_weight('bold')

                ax.legend(
                    wedges,
                    sorted_data['emotion'],
                    title="Emotions",
                    loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1)
                )

                ax.axis('equal')  

                plt.tight_layout()
                
                st.pyplot(fig)

        if has_confidence:
            with st.container(border=True):
                st.subheader("Confidence Analysis")

                confidence_data = date_filtered_df.groupby('emotion')['confidence_score'].mean().reset_index()
                confidence_data = confidence_data.sort_values('confidence_score', ascending=False)

                st.write("Average confidence score by emotion:")
                st.bar_chart(confidence_data.set_index('emotion'))

                try:
                    detailed_confidence = date_filtered_df.groupby('emotion').agg({
                        'count': 'sum',
                        'confidence_score': ['mean', 'min', 'max', 'std']
                    }).reset_index()

                    detailed_confidence.columns = ['Emotion', 'Count', 'Avg Confidence', 'Min Confidence', 'Max Confidence', 'Std Deviation']

                    for col in ['Avg Confidence', 'Min Confidence', 'Max Confidence', 'Std Deviation']:
                        detailed_confidence[col] = detailed_confidence[col].round(3)

                    st.dataframe(detailed_confidence.sort_values('Avg Confidence', ascending=False), use_container_width=True)
                except Exception as e:
                    print(f"Error creating detailed confidence table: {e}")
                    simple_confidence = pd.DataFrame({
                        'Emotion': confidence_data['emotion'],
                        'Avg Confidence': confidence_data['confidence_score'].round(3)
                    })
                    st.dataframe(simple_confidence, use_container_width=True)
        
        with st.container(border=True):
            st.subheader("Emotion Trends Over Time")

            time_data = date_filtered_df.copy()
            time_data['date'] = time_data['timestamp'].dt.date
            
            daily_emotion = time_data.groupby(['date', 'emotion'])['count'].sum().reset_index()

            daily_pivot = daily_emotion.pivot(index='date', columns='emotion', values='count').fillna(0)
            
            st.line_chart(daily_pivot)

        if has_confidence:
            try:
                with st.container(border=True):
                    st.subheader("Confidence Trends Over Time")

                    confidence_time = time_data.groupby(['date', 'emotion'])['confidence_score'].mean().reset_index()

                    confidence_pivot = confidence_time.pivot(index='date', columns='emotion', values='confidence_score').fillna(0)
                    
                    st.line_chart(confidence_pivot)
            except Exception as e:
                print(f"Error creating confidence trends: {e}")

        if len(daily_pivot) >= 7:
            with st.container(border=True):
                st.subheader("Weekly Emotion Patterns")
                time_data['day'] = time_data['timestamp'].dt.day_name()
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                weekly_emotion = time_data.groupby(['day', 'emotion'])['count'].sum().reset_index()

                weekly_pivot = weekly_emotion.pivot(index='day', columns='emotion', values='count').fillna(0)

                try:
                    weekly_pivot = weekly_pivot.reindex(day_order)
                except Exception as e:
                    print(f"Error reordering days: {e}")
                
                st.write("Total emotion counts by day of week")
                st.dataframe(weekly_pivot.style.highlight_max(axis=1, color='lightgreen').highlight_min(axis=1, color='#ffcccc'), use_container_width=True)

                if has_confidence:
                    try:
                        st.write("Average confidence by day of week")
                        weekly_confidence = time_data.groupby(['day', 'emotion'])['confidence_score'].mean().reset_index()
                        confidence_pivot = weekly_confidence.pivot(index='day', columns='emotion', values='confidence_score').fillna(0)
                        try:
                            confidence_pivot = confidence_pivot.reindex(day_order)
                        except Exception as e:
                            print(f"Error reordering days for confidence: {e}")

                        st.dataframe(confidence_pivot.style.highlight_max(axis=1, color='lightgreen').highlight_min(axis=1, color='#ffcccc'), use_container_width=True)
                    except Exception as e:
                        print(f"Error creating weekly confidence analysis: {e}")

                st.subheader("Emotions by Day of Week")

                day_totals = weekly_pivot.sum(axis=1).reset_index()
                day_totals.columns = ['Day', 'Total']
                
                st.bar_chart(day_totals.set_index('Day'))

        with st.container(border=True):
            st.subheader("Export Data")
            
            if st.button("Export to CSV"):
                if user_id:
                    export_filename = f"emotion_data_{st.session_state.get('username', user_id)}"
                else:
                    export_filename = "emotion_data_all"
                
                if session_option != "All Sessions":
                    export_filename += f"_{session_option}"
                
                export_filename += ".csv"
                date_filtered_df.to_csv(export_filename, index=False)
                
                st.success(f"Data exported to '{export_filename}'")

                with open(export_filename, 'rb') as f:
                    st.download_button(
                        label="Download CSV File",
                        data=f,
                        file_name=export_filename,
                        mime='text/csv'
                    )
    else:
        st.info("No data available for the selected filters.")

if __name__ == "__main__":
    analytics_page()