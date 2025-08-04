import cv2
import numpy as np
import tensorflow as tf
import pickle
from collections import deque
from googletrans import Translator
from hand_detection import HandDetector

class ASLPredictor:
    def __init__(self, model_path="asl_model.h5", encoder_path="label_encoder.pkl"):
        """Initialize ASL predictor"""
        self.detector = HandDetector()
        self.translator = Translator()
        
        # Load trained model
        print("📥 Loading trained model...")
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return
        
        # Load label encoder
        try:
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print("✅ Label encoder loaded!")
        except Exception as e:
            print(f"❌ Error loading label encoder: {e}")
            return
        
        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=10)  # Store last 10 predictions
        self.confidence_threshold = 0.7
        
        # Language settings
        self.target_languages = {
            'English': 'en',
            'Hindi': 'hi',
            'Tamil': 'ta',
            'Telugu': 'te',
            'Bengali': 'bn',
            'Marathi': 'mr',
            'Gujarati': 'gu',
            'Kannada': 'kn',
            'Malayalam': 'ml',
            'Punjabi': 'pa'
        }
        self.current_language = 'English'
        
        # Text storage
        self.recognized_text = ""
        self.translated_text = ""
        
    def extract_features(self, lm_list, img_shape):
        """Extract features from landmarks (same as training)"""
        features = []
        h, w, _ = img_shape
        
        for landmark in lm_list:
            x_norm = landmark[1] / w
            y_norm = landmark[2] / h
            features.extend([x_norm, y_norm])
        
        while len(features) < 42:
            features.extend([0.0, 0.0])
        
        return np.array(features[:42])
    
    def predict_gesture(self, features):
        """Predict gesture from features"""
        # Make prediction
        prediction = self.model.predict(features.reshape(1, -1), verbose=0)[0]
        
        # Get predicted class and confidence
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        
        # Get gesture letter
        gesture = self.label_encoder.classes_[predicted_class]
        
        return gesture, confidence
    
    def smooth_predictions(self, gesture, confidence):
        """Smooth predictions using buffer"""
        if confidence > self.confidence_threshold:
            self.prediction_buffer.append(gesture)
        
        if len(self.prediction_buffer) > 0:
            # Get most common prediction in buffer
            unique, counts = np.unique(self.prediction_buffer, return_counts=True)
            most_common = unique[np.argmax(counts)]
            return most_common
        
        return None
    
    def translate_text(self, text):
        """Translate text to target language"""
        if self.current_language == 'English' or not text:
            return text
        
        try:
            target_lang = self.target_languages[self.current_language]
            translated = self.translator.translate(text, dest=target_lang)
            return translated.text
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def run_real_time(self, camera_index=0):
        """Run real-time ASL recognition"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_index}")
            return
        
        print("🚀 Real-time ASL Recognition Started!")
        print("Controls:")
        print("  SPACE - Add predicted letter to text")
        print("  BACKSPACE - Delete last letter")
        print("  ENTER - Translate text")
        print("  'c' - Clear all text")
        print("  'l' - Change language")
        print("  'q' - Quit")
        
        while True:
            ret, img = cap.read()
            if not ret:
                break
            
            # Flip for natural interaction
            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            
            # Find hands
            img = self.detector.find_hands(img, draw=True)
            lm_list = self.detector.find_position(img, draw=False)
            
            current_prediction = "No hand"
            confidence = 0.0
            
            if len(lm_list) > 0:
                # Extract features
                features = self.extract_features(lm_list, img.shape)
                
                # Make prediction
                gesture, conf = self.predict_gesture(features)
                
                # Smooth prediction
                smoothed_gesture = self.smooth_predictions(gesture, conf)
                
                if smoothed_gesture:
                    current_prediction = smoothed_gesture
                    confidence = conf
            
            # Create display overlay
            overlay = img.copy()
            
            # Draw background rectangles
            cv2.rectangle(overlay, (10, 10), (w-10, 120), (0, 0, 0), -1)  # Top panel
            cv2.rectangle(overlay, (10, h-150), (w-10, h-10), (0, 0, 0), -1)  # Bottom panel
            
            # Blend overlay
            img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
            
            # Display current prediction
            cv2.putText(img, f'Prediction: {current_prediction}', (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f'Confidence: {confidence:.2f}', (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(img, f'Language: {self.current_language}', (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Display recognized text
            cv2.putText(img, f'Text: {self.recognized_text}', (20, h-120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Display translated text
            if self.translated_text:
                cv2.putText(img, f'Translated: {self.translated_text}', (20, h-90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Display controls
            cv2.putText(img, 'SPACE: Add | ENTER: Translate | C: Clear | Q: Quit', 
                       (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('ASL Recognition', img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space - add letter
                if current_prediction != "No hand" and confidence > self.confidence_threshold:
                    self.recognized_text += current_prediction
                    print(f"Added: {current_prediction} -> Text: {self.recognized_text}")
            
            elif key == 8:  # Backspace
                if self.recognized_text:
                    self.recognized_text = self.recognized_text[:-1]
                    self.translated_text = ""
                    print(f"Deleted -> Text: {self.recognized_text}")
            
            elif key == 13:  # Enter - translate
                if self.recognized_text:
                    self.translated_text = self.translate_text(self.recognized_text)
                    print(f"Translated to {self.current_language}: {self.translated_text}")
            
            elif key == ord('c'):  # Clear
                self.recognized_text = ""
                self.translated_text = ""
                print("Text cleared")
            
            elif key == ord('l'):  # Change language
                self.change_language()
            
            elif key == ord('q'):  # Quit
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def change_language(self):
        """Change target language"""
        print("\nAvailable languages:")
        for i, lang in enumerate(self.target_languages.keys(), 1):
            print(f"{i}. {lang}")
        
        try:
            choice = int(input("Choose language (number): "))
            languages = list(self.target_languages.keys())
            if 1 <= choice <= len(languages):
                self.current_language = languages[choice-1]
                print(f"Language changed to: {self.current_language}")
                
                # Re-translate existing text
                if self.recognized_text:
                    self.translated_text = self.translate_text(self.recognized_text)
            else:
                print("Invalid choice!")
        except ValueError:
            print("Invalid input!")

def main():
    """Main function"""
    print("🤖 ASL Real-time Recognition System")
    print("=" * 40)
    
    # Check if model files exist
    if not os.path.exists("asl_model.h5"):
        print("❌ Model file not found! Please train the model first:")
        print("   Run: python model_trainer.py")
        return
    
    if not os.path.exists("label_encoder.pkl"):
        print("❌ Label encoder not found! Please train the model first.")
        return
    
    # Initialize predictor
    predictor = ASLPredictor()
    
    # Get camera index
    camera_index = int(input("Camera index (0, 1, or 2): ") or 0)
    
    # Start real-time recognition
    predictor.run_real_time(camera_index)

if __name__ == "__main__":
    import os
    main()