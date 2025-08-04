import cv2
import numpy as np
import tensorflow as tf
import pickle
from collections import deque
from googletrans import Translator
import os
import sys

# Import hand detection
try:
    from hand_detection import HandDetector
except ImportError:
    print("❌ hand_detection.py not found! Please ensure it's in the same directory.")
    sys.exit(1)

class FixedUIASLPredictor:
    def __init__(self):
        """Initialize ASL predictor with better file handling"""
        
        print("🚀 Initializing ASL Predictor...")
        
        # Initialize hand detector
        self.detector = HandDetector()
        self.translator = Translator()
        
        # Try to load models in order of preference
        model_files = [
            ("improved_asl_model.h5", "improved_label_encoder.pkl", "feature_scaler.pkl"),
            ("best_model.h5", "label_encoder.pkl", None),
            ("asl_model.h5", "label_encoder.pkl", None)
        ]
        
        self.model = None
        self.label_encoder = None
        self.scaler = None
        
        for model_path, encoder_path, scaler_path in model_files:
            if os.path.exists(model_path) and os.path.exists(encoder_path):
                try:
                    # Load model
                    print(f"📥 Loading model: {model_path}")
                    self.model = tf.keras.models.load_model(model_path)
                    print("✅ Model loaded successfully!")
                    
                    # Load label encoder
                    print(f"📥 Loading encoder: {encoder_path}")
                    with open(encoder_path, 'rb') as f:
                        self.label_encoder = pickle.load(f)
                    print("✅ Label encoder loaded!")
                    
                    # Try to load scaler if available
                    if scaler_path and os.path.exists(scaler_path):
                        print(f"📥 Loading scaler: {scaler_path}")
                        with open(scaler_path, 'rb') as f:
                            self.scaler = pickle.load(f)
                        print("✅ Feature scaler loaded!")
                    else:
                        print("⚠️ No feature scaler found - using raw features")
                    
                    break
                    
                except Exception as e:
                    print(f"❌ Error loading {model_path}: {e}")
                    continue
        
        if self.model is None or self.label_encoder is None:
            print("❌ Could not load any valid model! Please train a model first.")
            sys.exit(1)
        
        # Prediction settings
        self.prediction_buffer = deque(maxlen=5)
        self.confidence_threshold = 0.8
        
        # Stability tracking
        self.stable_prediction = None
        self.stable_count = 0
        self.stability_threshold = 3
        
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
        
        print("🎯 Predictor initialized successfully!")
    
    def extract_features(self, lm_list, img_shape):
        """Extract features from landmarks"""
        if len(lm_list) == 0:
            return None
        
        try:
            features = []
            h, w, _ = img_shape
            
            # Extract normalized coordinates
            for landmark in lm_list:
                x_norm = landmark[1] / w
                y_norm = landmark[2] / h
                features.extend([x_norm, y_norm])
            
            # Ensure we have exactly 42 features
            while len(features) < 42:
                features.extend([0.0, 0.0])
            
            features = features[:42]
            features_array = np.array(features, dtype=np.float32)
            
            # Check for invalid values
            if np.isnan(features_array).any() or np.isinf(features_array).any():
                return None
            
            # Apply feature scaling if available
            if self.scaler is not None:
                features_array = self.scaler.transform(features_array.reshape(1, -1))[0]
            
            return features_array
            
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return None
    
    def predict_gesture(self, features):
        """Predict gesture from features"""
        try:
            # Make prediction
            prediction = self.model.predict(features.reshape(1, -1), verbose=0)[0]
            
            # Get predicted class and confidence
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]
            
            # Get gesture letter
            gesture = self.label_encoder.classes_[predicted_class]
            
            return gesture, confidence
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return None, 0.0
    
    def stabilize_prediction(self, gesture, confidence):
        """Stabilize predictions for consistent recognition"""
        if confidence < self.confidence_threshold:
            return None, False
        
        # Check if this is the same as our current stable prediction
        if gesture == self.stable_prediction:
            self.stable_count += 1
        else:
            self.stable_prediction = gesture
            self.stable_count = 1
        
        # Return stable prediction if we have enough consistent predictions
        is_stable = self.stable_count >= self.stability_threshold
        return gesture if is_stable else None, is_stable
    
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
    
    def draw_ui_overlay(self, img, prediction, confidence, is_stable, fps=0):
        """Draw user interface overlay with bright, visible colors"""
        h, w, _ = img.shape
        
        # Create a more opaque overlay for better visibility
        overlay = img.copy()
        
        # Top panel - make it bigger and more visible
        cv2.rectangle(overlay, (10, 10), (w-10, 150), (0, 0, 0), -1)
        # Bottom panel
        cv2.rectangle(overlay, (10, h-200), (w-10, h-10), (0, 0, 0), -1)
        
        # Blend with higher opacity
        img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        
        # Add white borders for better visibility
        cv2.rectangle(img, (10, 10), (w-10, 150), (255, 255, 255), 2)
        cv2.rectangle(img, (10, h-200), (w-10, h-10), (255, 255, 255), 2)
        
        # Current prediction with larger font and bright colors
        pred_color = (0, 255, 0) if is_stable else (0, 165, 255)  # Green if stable, orange if not
        cv2.putText(img, f'Prediction: {prediction}', (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, pred_color, 3)
        
        # Confidence with color coding
        conf_color = (0, 255, 0) if confidence > self.confidence_threshold else (0, 0, 255)
        cv2.putText(img, f'Confidence: {confidence:.3f}', (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, conf_color, 2)
        
        # Stability indicator
        stability_text = f'Stable: {"YES" if is_stable else "NO"} ({self.stable_count}/{self.stability_threshold})'
        stability_color = (0, 255, 0) if is_stable else (0, 255, 255)
        cv2.putText(img, stability_text, (20, 115), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, stability_color, 2)
        
        # Language and threshold info
        cv2.putText(img, f'Language: {self.current_language} | Threshold: {self.confidence_threshold}', 
                   (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Recognized text with word wrapping
        if len(self.recognized_text) > 40:
            # Split long text into multiple lines
            text_line1 = f'Text: {self.recognized_text[:40]}'
            text_line2 = f'      {self.recognized_text[40:80]}'
            cv2.putText(img, text_line1, (20, h-170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if len(self.recognized_text) > 40:
                cv2.putText(img, text_line2, (20, h-145), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.putText(img, f'Text: {self.recognized_text}', (20, h-170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Translated text
        if self.translated_text:
            if len(self.translated_text) > 50:
                trans_line1 = f'Trans: {self.translated_text[:50]}'
                trans_line2 = f'       {self.translated_text[50:100]}'
                cv2.putText(img, trans_line1, (20, h-120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if len(self.translated_text) > 50:
                    cv2.putText(img, trans_line2, (20, h-95), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(img, f'Translated: {self.translated_text}', (20, h-120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Enhanced controls with better visibility
        controls = [
            'SPACE: Add Letter | ENTER: Translate | C: Clear | Q: Quit',
            'BACKSPACE: Delete | L: Language | +/-: Adjust Threshold'
        ]
        cv2.putText(img, controls[0], (20, h-60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        cv2.putText(img, controls[1], (20, h-35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        
        # FPS counter in top right
        if fps > 0:
            cv2.putText(img, f'FPS: {fps:.1f}', (w-120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return img
    
    def run_real_time(self, camera_index=0):
        """Run real-time ASL recognition with fixed UI"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_index}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("🚀 Real-time ASL Recognition Started!")
        print("\n📋 Controls:")
        print("  SPACE    - Add predicted letter to text")
        print("  BACKSPACE- Delete last letter")
        print("  ENTER    - Translate text")
        print("  'c'      - Clear all text")
        print("  'l'      - Change language")
        print("  '+'      - Increase confidence threshold")
        print("  '-'      - Decrease confidence threshold")
        print("  'q'      - Quit")
        
        frame_count = 0
        fps_counter = deque(maxlen=30)
        
        while True:
            ret, img = cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            frame_count += 1
            start_time = cv2.getTickCount()
            
            # Flip for natural interaction
            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            
            # Find hands
            img = self.detector.find_hands(img, draw=True)
            lm_list = self.detector.find_position(img, draw=False)
            
            # Initialize prediction variables
            current_prediction = "No hand detected"
            confidence = 0.0
            is_stable = False
            
            if len(lm_list) > 0:
                # Extract features
                features = self.extract_features(lm_list, img.shape)
                
                if features is not None:
                    # Make prediction
                    gesture, conf = self.predict_gesture(features)
                    
                    if gesture is not None:
                        # Stabilize prediction
                        stable_gesture, is_stable = self.stabilize_prediction(gesture, conf)
                        
                        current_prediction = stable_gesture if stable_gesture else f"{gesture} (unstable)"
                        confidence = conf
                else:
                    current_prediction = "Feature extraction failed"
            
            # Calculate FPS
            end_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (end_time - start_time)
            fps_counter.append(fps)
            avg_fps = np.mean(fps_counter)
            
            # Draw UI overlay
            img = self.draw_ui_overlay(img, current_prediction, confidence, is_stable, avg_fps)
            
            cv2.imshow('ASL Recognition - Fixed UI', img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space - add letter
                if current_prediction not in ["No hand detected", "Feature extraction failed"] and is_stable:
                    clean_prediction = current_prediction.split(" (")[0]
                    self.recognized_text += clean_prediction
                    print(f"✅ Added: {clean_prediction} -> Text: '{self.recognized_text}'")
                    self.stable_prediction = None
                    self.stable_count = 0
                else:
                    print(f"⚠️ Cannot add: prediction='{current_prediction}', stable={is_stable}")
            
            elif key == 8:  # Backspace
                if self.recognized_text:
                    removed = self.recognized_text[-1]
                    self.recognized_text = self.recognized_text[:-1]
                    self.translated_text = ""
                    print(f"🗑️ Deleted '{removed}' -> Text: '{self.recognized_text}'")
            
            elif key == 13:  # Enter - translate
                if self.recognized_text:
                    self.translated_text = self.translate_text(self.recognized_text)
                    print(f"🌐 Translated to {self.current_language}: '{self.translated_text}'")
            
            elif key == ord('c'):  # Clear
                self.recognized_text = ""
                self.translated_text = ""
                self.stable_prediction = None
                self.stable_count = 0
                print("🧹 Text cleared")
            
            elif key == ord('l'):  # Change language
                self.change_language()
            
            elif key == ord('+') or key == ord('='):  # Increase threshold
                self.confidence_threshold = min(0.99, self.confidence_threshold + 0.05)
                print(f"🎯 Confidence threshold increased to: {self.confidence_threshold:.2f}")
            
            elif key == ord('-'):  # Decrease threshold
                self.confidence_threshold = max(0.50, self.confidence_threshold - 0.05)
                print(f"🎯 Confidence threshold decreased to: {self.confidence_threshold:.2f}")
            
            elif key == ord('q'):  # Quit
                print("👋 Goodbye!")
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def change_language(self):
        """Change target language"""
        print("\n🌐 Available languages:")
        for i, lang in enumerate(self.target_languages.keys(), 1):
            marker = "👉" if lang == self.current_language else "  "
            print(f"{marker} {i}. {lang}")
        
        try:
            choice = input("\nChoose language (number): ").strip()
            if choice.isdigit():
                choice = int(choice)
                languages = list(self.target_languages.keys())
                if 1 <= choice <= len(languages):
                    self.current_language = languages[choice-1]
                    print(f"✅ Language changed to: {self.current_language}")
                    
                    if self.recognized_text:
                        self.translated_text = self.translate_text(self.recognized_text)
                        print(f"🔄 Re-translated: '{self.translated_text}'")
                else:
                    print("❌ Invalid choice!")
            else:
                print("❌ Please enter a number!")
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function"""
    print("🤖 ASL Real-time Recognition System - Fixed UI")
    print("=" * 50)
    
    # Initialize predictor
    try:
        predictor = FixedUIASLPredictor()
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        return
    
    # Get camera index
    try:
        camera_input = input("Camera index (0 for default): ").strip()
        camera_index = int(camera_input) if camera_input.isdigit() else 0
    except ValueError:
        print("Using default camera (index 0)")
        camera_index = 0
    
    # Start real-time recognition
    predictor.run_real_time(camera_index)

if __name__ == "__main__":
    main()