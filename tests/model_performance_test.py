import cv2
import torch
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
import sys
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Face_Emotion_CNN

class EmotionDetectionPerformanceAnalyzer:
    def __init__(self, model_path='./models/FER_trained_model.pt', 
                 cascade_path='./models/haarcascade_frontalface_default.xml',
                 use_gpu=True):

        self.model_path = model_path
        self.cascade_path = cascade_path
        self.use_gpu = use_gpu
        self.model = None
        self.face_cascade = None
        self.emotion_dict = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness',
                            4: 'anger', 5: 'disgust', 6: 'fear'}

        self.frame_times = []
        self.face_detection_times = []
        self.emotion_prediction_times = []
        self.total_faces_detected = 0
        self.frames_processed = 0
        self.faces_per_frame = []

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.load_model()
        self.load_cascade()
    
    def load_model(self):
        start_time = time.time()
        
        try:
            self.model = Face_Emotion_CNN()

            if not os.path.exists(self.model_path):
                print(f"ERROR: Model file not found at {self.model_path}")
                return False

            self.model.load_state_dict(torch.load(self.model_path, 
                                        map_location=lambda storage, loc: storage), 
                                      strict=False)

            if self.use_gpu and torch.cuda.is_available():
                self.model = self.model.cuda()
                print("Model loaded on GPU")
            else:
                print("Model loaded on CPU")

            self.model.eval()
            
            model_load_time = time.time() - start_time
            print(f"Model loaded in {model_load_time:.4f} seconds")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def load_cascade(self):
        try:
            self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
            
            if self.face_cascade.empty():
                print(f"ERROR: Face cascade file not found or invalid at {self.cascade_path}")
                return False
                
            print("Face cascade classifier loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading face cascade: {e}")
            return False
    
    def process_frame(self, frame, scale_factor=1.2, min_neighbors=5, min_size=(30, 30), 
                     record_metrics=True):

        if self.model is None or self.face_cascade is None:
            print("Model or face cascade not loaded")
            return frame, []
        
        frame_start_time = time.time()
        emotions_detected = []
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        face_detection_start = time.time()
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        face_detection_time = time.time() - face_detection_start
        
        if record_metrics:
            self.face_detection_times.append(face_detection_time)
            self.faces_per_frame.append(len(faces))
            self.total_faces_detected += len(faces)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            emotion_prediction_start = time.time()
            
            try:
                resize_frame = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
                X = resize_frame/256 
                X = Image.fromarray(X)
                X = self.transform(X).unsqueeze(0)

                if self.use_gpu and torch.cuda.is_available():
                    X = X.cuda()

                with torch.no_grad():
                    log_ps = self.model(X)
                    ps = torch.exp(log_ps)

                    if torch.cuda.is_available():
                        top_p, top_class = ps.cpu().topk(1, dim=1)
                        predicted_emotion = self.emotion_dict[int(top_class.numpy())]
                        confidence = float(top_p.numpy()[0][0])
                    else:
                        top_p, top_class = ps.topk(1, dim=1)
                        predicted_emotion = self.emotion_dict[int(top_class.numpy())]
                        confidence = float(top_p.numpy()[0][0])
                
                emotion_prediction_time = time.time() - emotion_prediction_start
                
                if record_metrics:
                    self.emotion_prediction_times.append(emotion_prediction_time)

                cv2.putText(frame, f"{predicted_emotion} ({confidence:.2f})", 
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (0, 255, 0), 2)
                
                emotions_detected.append({
                    'emotion': predicted_emotion,
                    'confidence': confidence,
                    'position': (x, y, w, h)
                })
                
            except Exception as e:
                print(f"Error in emotion prediction: {e}")

        frame_time = time.time() - frame_start_time
        
        if record_metrics:
            self.frame_times.append(frame_time)
            self.frames_processed += 1
        
        return frame, emotions_detected
    
    def analyze_webcam(self, duration=30, display=True, resolution_scale=1.0,
                      frame_skip=0):

        print(f"Starting webcam analysis for {duration} seconds...")

        self.frame_times = []
        self.face_detection_times = []
        self.emotion_prediction_times = []
        self.total_faces_detected = 0
        self.frames_processed = 0
        self.faces_per_frame = []

        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return None
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Could not read frame from webcam")
                    break
                frame_count += 1
                if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
                    continue
                if resolution_scale < 1.0:
                    width = int(frame.shape[1] * resolution_scale)
                    height = int(frame.shape[0] * resolution_scale)
                    frame = cv2.resize(frame, (width, height))

                processed_frame, emotions = self.process_frame(frame)

                if display:
                    cv2.imshow('Emotion Detection Performance Analysis', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

            total_time = time.time() - start_time
            metrics = self.calculate_metrics(total_time)

            self.print_performance_summary(metrics)
            
            return metrics
    
    def analyze_video(self, video_path, display=True, resolution_scale=1.0,
                     frame_skip=0, max_frames=None):

        print(f"Starting video analysis for {video_path}...")
        
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return None

        self.frame_times = []
        self.face_detection_times = []
        self.emotion_prediction_times = []
        self.total_faces_detected = 0
        self.frames_processed = 0
        self.faces_per_frame = []

        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video file")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        print(f"Video info: {frame_count} frames, {fps:.2f} FPS, {duration:.2f} seconds")
        
        if max_frames is not None and max_frames < frame_count:
            frame_count = max_frames
            print(f"Limiting analysis to {max_frames} frames")
        
        start_time = time.time()
        frame_index = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret or (max_frames is not None and frame_index >= max_frames):
                    break
                frame_index += 1
                if frame_skip > 0 and frame_index % (frame_skip + 1) != 0:
                    continue
                if resolution_scale < 1.0:
                    width = int(frame.shape[1] * resolution_scale)
                    height = int(frame.shape[0] * resolution_scale)
                    frame = cv2.resize(frame, (width, height))

                processed_frame, emotions = self.process_frame(frame)

                if display:
                    cv2.imshow('Emotion Detection Performance Analysis', processed_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

            total_time = time.time() - start_time
            metrics = self.calculate_metrics(total_time)

            self.print_performance_summary(metrics)
            
            return metrics
    
    def calculate_metrics(self, total_time):
        metrics = {
            'total_time': total_time,
            'frames_processed': self.frames_processed,
            'total_faces_detected': self.total_faces_detected,
            'avg_fps': self.frames_processed / total_time if total_time > 0 else 0,
            'avg_frame_time': np.mean(self.frame_times) if self.frame_times else 0,
            'max_frame_time': np.max(self.frame_times) if self.frame_times else 0,
            'min_frame_time': np.min(self.frame_times) if self.frame_times else 0,
            'avg_face_detection_time': np.mean(self.face_detection_times) if self.face_detection_times else 0,
            'avg_emotion_prediction_time': np.mean(self.emotion_prediction_times) if self.emotion_prediction_times else 0,
            'avg_faces_per_frame': np.mean(self.faces_per_frame) if self.faces_per_frame else 0,
            'max_faces_per_frame': np.max(self.faces_per_frame) if self.faces_per_frame else 0,
        }

        if metrics['avg_frame_time'] > 0:
            metrics['face_detection_percent'] = (metrics['avg_face_detection_time'] / metrics['avg_frame_time']) * 100
            metrics['emotion_prediction_percent'] = (metrics['avg_emotion_prediction_time'] / metrics['avg_frame_time']) * 100
        else:
            metrics['face_detection_percent'] = 0
            metrics['emotion_prediction_percent'] = 0
        
        return metrics
    
    def print_performance_summary(self, metrics):
        """Print a summary of performance metrics"""
        print("\n" + "="*50)
        print("EMOTION DETECTION PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Total time: {metrics['total_time']:.2f} seconds")
        print(f"Frames processed: {metrics['frames_processed']}")
        print(f"Faces detected: {metrics['total_faces_detected']}")
        print(f"Average FPS: {metrics['avg_fps']:.2f}")
        print(f"Average faces per frame: {metrics['avg_faces_per_frame']:.2f}")
        print(f"Maximum faces in a single frame: {metrics['max_faces_per_frame']}")
        print("\nTiming Breakdown:")
        print(f"  Average frame processing time: {metrics['avg_frame_time']*1000:.2f} ms")
        print(f"  Average face detection time: {metrics['avg_face_detection_time']*1000:.2f} ms ({metrics['face_detection_percent']:.1f}%)")
        print(f"  Average emotion prediction time: {metrics['avg_emotion_prediction_time']*1000:.2f} ms ({metrics['emotion_prediction_percent']:.1f}%)")
        print(f"  Frame time range: {metrics['min_frame_time']*1000:.2f} - {metrics['max_frame_time']*1000:.2f} ms")
        print("="*50)
    
def main():
    parser = argparse.ArgumentParser(description='Emotion Detection Performance Analysis')
    parser.add_argument('--mode', choices=['webcam', 'video'], default='webcam', 
                        help='Analysis mode: webcam or video')
    parser.add_argument('--video', type=str, default=None, 
                        help='Path to video file (required for video mode)')
    parser.add_argument('--duration', type=int, default=30, 
                        help='Duration in seconds for webcam analysis')
    parser.add_argument('--max-frames', type=int, default=None, 
                        help='Maximum number of frames to process in video mode')
    parser.add_argument('--display', action='store_true', default=True, 
                        help='Display processed frames')
    parser.add_argument('--no-display', dest='display', action='store_false', 
                        help='Do not display processed frames')
    parser.add_argument('--scale', type=float, default=1.0, 
                        help='Resolution scale factor')
    parser.add_argument('--skip', type=int, default=0, 
                        help='Number of frames to skip between processing')
    parser.add_argument('--no-gpu', dest='use_gpu', action='store_false', default=True, 
                        help='Disable GPU acceleration')
    parser.add_argument('--model', type=str, default='./models/FER_trained_model.pt', 
                        help='Path to emotion detection model')
    parser.add_argument('--cascade', type=str, default='./models/haarcascade_frontalface_default.xml', 
                        help='Path to face cascade classifier')
    
    args = parser.parse_args()

    if args.mode == 'video' and args.video is None:
        parser.error("Video mode requires --video argument")

    analyzer = EmotionDetectionPerformanceAnalyzer(
        model_path=args.model,
        cascade_path=args.cascade,
        use_gpu=args.use_gpu
    )

    if args.mode == 'webcam':
        metrics = analyzer.analyze_webcam(
            duration=args.duration,
            display=args.display,
            resolution_scale=args.scale,
            frame_skip=args.skip
        )
    else: 
        metrics = analyzer.analyze_video(
            video_path=args.video,
            display=args.display,
            resolution_scale=args.scale,
            frame_skip=args.skip,
            max_frames=args.max_frames
        )

if __name__ == "__main__":
    main()