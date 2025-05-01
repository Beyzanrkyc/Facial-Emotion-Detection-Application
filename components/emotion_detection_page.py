import streamlit as st
import pandas as pd
import cv2
import torch
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import sys
from datetime import datetime
from utils.data_utils import save_emotion_data
from model import Face_Emotion_CNN

def emotion_detection_page():

    @st.cache_resource
    def load_model():
        model = Face_Emotion_CNN()
        model_path = './models/FER_trained_model.pt'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
            if torch.cuda.is_available():
                model = model.cuda()
                st.success("Using GPU acceleration!")
        else:
            st.warning("Model file not found.")
        return model

    emotion_dict = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness',
                    4: 'anger', 5: 'disgust', 6: 'fear'} 

    val_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    st.title("Real-time Emotion Detection")
    st.caption("Powered by OpenCV, PyTorch, and Streamlit")

    if 'authenticated' in st.session_state and st.session_state['authenticated']:
        if 'username' in st.session_state:
            st.success(f"Logged in as {st.session_state.get('username', 'User')}. Your emotion data will be saved to your account.")
    elif 'guest_mode' in st.session_state and st.session_state['guest_mode']:
        st.info("You're in guest mode. No data will be saved for privacy reasons.")

    with st.sidebar:
        st.title("Emotion Detection Controls")
        use_face_detection = st.checkbox("Enable Face Detection", value=True)
        use_display_emotion = st.checkbox("Display Emotion", value=True)
        show_confidence = st.checkbox("Show Confidence Score", value=True)

        st.title("Performance Settings")
        frame_skip = st.slider("Frame Skip", 0, 5, 2, help="Higher values improve performance but reduce responsiveness")
        resolution_scale = st.slider("Resolution Scale", 0.25, 1.0, 0.5, help="Lower values improve performance")

        st.title("Session Settings")

        user_identifier = "guest"
        if 'authenticated' in st.session_state and st.session_state['authenticated']:
            if 'username' in st.session_state:
                user_identifier = st.session_state['username']
            if 'user_id' in st.session_state:
                user_id = st.session_state['user_id']
            else:
                user_id = None
        else:
            user_id = None
        
        session_name = st.text_input("Session Name (optional)", 
                                     value=f"{user_identifier}_Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        with st.expander("Emotion Metrics"):
            emotion_metrics = {emotion: 0 for emotion in emotion_dict.values()}
            emotion_counters = st.columns(len(emotion_dict))
            emotion_displays = {}
            for i, (_, emotion) in enumerate(emotion_dict.items()):
                emotion_displays[emotion] = emotion_counters[i].metric(emotion, 0)

    if "stop" not in st.session_state:
        st.session_state.stop = False
    if "emotion_counts" not in st.session_state:
        st.session_state.emotion_counts = {emotion: 0 for emotion in emotion_dict.values()}
    if "emotion_confidence" not in st.session_state:
        st.session_state.emotion_confidence = {emotion: 0.0 for emotion in emotion_dict.values()}
    if "frame_count" not in st.session_state:
        st.session_state.frame_count = 0
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    try:
        model = load_model()
        model.eval()
        st.success("Emotion detection model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None

    col1, col2, col3, col4 = st.columns(4)
    start_button = col1.button("Start Webcam")
    stop_button = col2.button("Stop Webcam")
    reset_button = col3.button("Reset Metrics")

    if 'authenticated' in st.session_state and st.session_state['authenticated'] and not st.session_state.get('guest_mode', False):
        save_button = col4.button("Save Data")
    else:
        with col4:
            st.button("Save Data", disabled=True, help="Login to save your emotion data")
    
    if start_button:
        st.session_state.stop = False
        if st.session_state.session_id is None:
            st.session_state.session_id = session_name

    if stop_button:
        st.session_state.stop = True
        
    if reset_button:
        st.session_state.emotion_counts = {emotion: 0 for emotion in emotion_dict.values()}
        st.session_state.emotion_confidence = {emotion: 0.0 for emotion in emotion_dict.values()}
        for emotion in emotion_dict.values():
            emotion_displays[emotion].metric(emotion, 0)

    if 'authenticated' in st.session_state and st.session_state['authenticated'] and not st.session_state.get('guest_mode', False):
        if 'save_button' in locals() and save_button:
            if sum(st.session_state.emotion_counts.values()) > 0:
                saved_session_id = save_emotion_data(
                    st.session_state.emotion_counts, 
                    user_id=user_id, 
                    session_id=session_name
                )
                st.success(f"Data saved successfully! Session ID: {saved_session_id}")
                st.session_state.emotion_counts = {emotion: 0 for emotion in emotion_dict.values()}
                st.session_state.emotion_confidence = {emotion: 0.0 for emotion in emotion_dict.values()}
                for emotion in emotion_dict.values():
                    emotion_displays[emotion].metric(emotion, 0)
            else:
                st.warning("No emotion data to save. Start detecting emotions first!")

    def emotion_detection():
        frame_placeholder = st.empty()
        info_placeholder = st.empty()

        try:
            face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
            if face_cascade.empty():
                st.warning("Face detection model not found. Using webcam without face detection.")
                face_cascade = None
        except Exception as e:
            st.error(f"Error loading face detection model: {e}")
            face_cascade = None

        cap = cv2.VideoCapture(0)

        display_update_rate = max(1, frame_skip)  
        
        while cap.isOpened() and not st.session_state.stop:
            ret, frame = cap.read()
            if not ret:
                st.write("Video Capture Ended")
                break

            st.session_state.frame_count += 1

            if frame_skip > 0 and st.session_state.frame_count % (frame_skip + 1) != 0:
                continue

            if resolution_scale < 1.0:
                width = int(frame.shape[1] * resolution_scale)
                height = int(frame.shape[0] * resolution_scale)
                frame = cv2.resize(frame, (width, height))

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            detected_emotion = None
            confidence_score = 0.0

            if use_face_detection and face_cascade is not None and model is not None:
                # Optimising face detection parameters
                faces = face_cascade.detectMultiScale(
                    gray,  # Use gray for detection (faster)
                    scaleFactor=1.2,  # for faster processing
                    minNeighbors=5,   # for false positives
                    minSize=(30, 30)  # for larger than 30x30 pixel images
                )
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)

                    resize_frame = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
                    X = resize_frame/256 

                    X = Image.fromarray(X)
                    X = val_transform(X).unsqueeze(0)

                    if torch.cuda.is_available():
                        X = X.cuda()

                    with torch.no_grad():
                        model.eval()
                        log_ps = model(X)  
                        ps = torch.exp(log_ps)

                        if torch.cuda.is_available():
                            top_p, top_class = ps.cpu().topk(1, dim=1)
                            predicted_emotion = emotion_dict[int(top_class.numpy())]
                            confidence_score = float(top_p.numpy()[0][0])  
                        else:
                            top_p, top_class = ps.topk(1, dim=1)
                            predicted_emotion = emotion_dict[int(top_class.numpy())]
                            confidence_score = float(top_p.numpy()[0][0]) 

                        st.session_state.emotion_counts[predicted_emotion] += 1
                        st.session_state.emotion_confidence[predicted_emotion] = confidence_score
                        
                        detected_emotion = predicted_emotion

                        if use_display_emotion:
                            emotion_text = predicted_emotion
                            if show_confidence:
                                emotion_text += f" ({confidence_score:.2f})"
                            cv2.putText(frame_rgb, emotion_text, 
                                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.8, (0, 255, 0), 2)  

            if st.session_state.frame_count % display_update_rate == 0:
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                for emotion, count in st.session_state.emotion_counts.items():
                    emotion_displays[emotion].metric(emotion, count)

                if detected_emotion:
                    info_text = f"Detected emotion: {detected_emotion}"
                    if show_confidence:
                        info_text += f" (Confidence: {confidence_score:.2f})"
                    info_placeholder.info(info_text)

        cap.release()

    def display_metrics():
        st.subheader("Current Session Emotion Distribution")
        if sum(st.session_state.emotion_counts.values()) > 0:
            emotion_data = pd.DataFrame({
                'Emotion': list(st.session_state.emotion_counts.keys()),
                'Count': list(st.session_state.emotion_counts.values()),
                'Confidence': [st.session_state.emotion_confidence.get(emotion, 0.0) 
                              for emotion in st.session_state.emotion_counts.keys()]
            })

            emotion_data = emotion_data.sort_values('Count', ascending=False)

            st.bar_chart(emotion_data.set_index('Emotion')['Count'])

            if show_confidence:
                st.subheader("Confidence Scores")
                st.dataframe(
                    emotion_data[['Emotion', 'Confidence']].set_index('Emotion'),
                    use_container_width=True
                )
        else:
            st.info("Start the webcam to collect emotion data")

    if not st.session_state.stop:
        emotion_detection()

    display_metrics()

    with st.expander("About this app"):
        st.markdown("""
        ## Emotion Detection App
        This application uses a deep learning model to detect facial emotions in real-time.
        
        ### How it works:
        1. The webcam captures your face
        2. A face detection algorithm locates faces in the frame
        3. Each detected face is processed by a CNN model that classifies emotions
        4. The detected emotion is displayed on the screen and counted in metrics
        
        ### Performance tips:
        - Increase Frame Skip for better performance
        - Decrease Resolution Scale for smoother operation
        - GPU acceleration is automatically used when available
        
        ### Supported emotions:
        - Neutral
        - Happiness
        - Surprise
        - Sadness
        - Anger
        - Disgust
        - Fear
        
        ### Privacy:
        - When logged in, your emotion data is saved to your account
        - In guest mode, no data is saved for privacy
        """)

if __name__ == "__main__":
    emotion_detection_page()