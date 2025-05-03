import unittest
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import Face_Emotion_CNN

class TestFacialEmotionDetection(unittest.TestCase):
    
    def setUp(self):
        self.model = Face_Emotion_CNN()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507395516207, ),(0.255128989415, ))
        ])
        self.emotion_dict = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness',
                             4: 'anger', 5: 'disgust', 6: 'fear'}

        self.face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
        self.test_image = np.zeros((48, 48), dtype=np.uint8)
        
    def test_model_initialization(self):
        self.assertIsInstance(self.model, Face_Emotion_CNN)
        self.assertEqual(self.model.cnn1.in_channels, 1, "Input channels should be 1 for grayscale")
        self.assertEqual(self.model.fc3.out_features, 7, "Output features should be 7 for emotions")
        print("[Model Init] Result: CNN structure verified - 1 input channel, 7 emotion outputs")
    
    def test_model_forward_pass(self):
        x = torch.randn(1, 1, 48, 48)

        self.model.eval()
        with torch.no_grad():
            output = self.model(x)

        self.assertEqual(output.shape, (1, 7), "Output shape should be [1, 7]")

        probabilities = torch.exp(output)
        sum_prob = float(torch.sum(probabilities))
        self.assertAlmostEqual(sum_prob, 1.0, places=5)
        print(f"[Forward Pass] Result: Output shape correct [1,7], probability sum: {sum_prob:.5f}")
    
    def test_face_detection(self):
        if self.face_cascade.empty():
            self.skipTest("Face cascade classifier not loaded")

        test_image = np.ones((100, 100), dtype=np.uint8) * 128
        cv2.circle(test_image, (50, 50), 30, 255, -1)  
        cv2.circle(test_image, (40, 40), 5, 0, -1)    
        cv2.circle(test_image, (60, 40), 5, 0, -1)     
        cv2.ellipse(test_image, (50, 60), (15, 10), 0, 0, 180, 0, -1) 
        
        faces = self.face_cascade.detectMultiScale(
            test_image,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30)
        )

        self.assertTrue(isinstance(faces, tuple) or isinstance(faces, np.ndarray))
        print(f"[Face Detection] Result: Detection function returned {type(faces).__name__} with {len(faces)} potential faces")
    
    def test_image_preprocessing(self):
        pil_image = Image.fromarray(self.test_image)

        tensor_image = self.transform(pil_image)

        self.assertEqual(tensor_image.shape, (1, 48, 48), "Transformed image should be [1, 48, 48]")
        self.assertIsInstance(tensor_image, torch.Tensor)
        print(f"[Preprocessing] Result: Image correctly transformed to tensor shape {tuple(tensor_image.shape)}")
    
    def test_end_to_end_prediction(self):
        pil_image = Image.fromarray(self.test_image)
        tensor_image = self.transform(pil_image).unsqueeze(0)  

        self.model.eval()
        with torch.no_grad():
            log_ps = self.model(tensor_image)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)

            top_class_value = top_class.item()
            predicted_emotion = self.emotion_dict[top_class_value]
            confidence_score = float(top_p.item())

        self.assertIn(predicted_emotion, self.emotion_dict.values())
        self.assertTrue(0.0 <= confidence_score <= 1.0)
        print(f"[Prediction] Result: Predicted '{predicted_emotion}' with {confidence_score:.4f} confidence")
    
    def test_model_parameter_count(self):
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.assertTrue(100000 < param_count < 10000000)
        print(f"[Parameters] Result: Model has {param_count:,} trainable parameters")
        
if __name__ == '__main__':
    print("Running Facial Emotion Detection Tests...")
    unittest.main(verbosity=1)