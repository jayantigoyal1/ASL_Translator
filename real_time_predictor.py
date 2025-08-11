import cv2
import numpy as np
import tensorflow as tf
import pickle
from collections import deque
from googletrans import Translator
import os
import sys
from PIL import Image, ImageDraw, ImageFont

# Import hand detection
try:
    from hand_detection import HandDetector
except ImportError:
    print("❌ hand_detection.py not found! Please ensure it's in the same directory.")
    sys.exit(1)

class ImprovedASLPredictor:
    def __init__(self, model_path="improved_asl_model.h5", 
                 encoder_path="improved_label_encoder.pkl",
                 scaler_path="feature_scaler.pkl"):
        """Initialize improved ASL predictor"""
        
        print("🚀 Initializing Improved ASL Predictor...")
        
        # Initialize hand detector
        self.detector = HandDetector()
        self.translator = Translator()
        
        # Check if all required files exist
        required_files = [model_path, encoder_path, scaler_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            print("Please run the improved model trainer first!")
            return
        
        # Load trained model
        print("📥 Loading improved model...")
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return
        
        # Load label encoder
        print("📥 Loading label encoder...")
        try:
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print("✅ Label encoder loaded!")
        except Exception as e:
            print(f"❌ Error loading label encoder: {e}")
            return
        
        # Load feature scaler
        print("📥 Loading feature scaler...")
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("✅ Feature scaler loaded!")
        except Exception as e:
            print(f"❌ Error loading scaler: {e}")
            self.scaler = None
        
        # Prediction settings
        self.prediction_buffer = deque(maxlen=7)  # Reduced buffer for faster response
        self.confidence_threshold = 0.85  # Higher threshold for better accuracy
        self.min_detection_confidence = 0.5
        
        # Stability tracking
        self.stable_prediction = None
        self.stable_count = 0
        self.stability_threshold = 3  # Need 3 consistent predictions
        
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
        
        # Debug mode
        self.debug_mode = False
        
        print("🎯 Predictor initialized successfully!")
        print(f"   Confidence threshold: {self.confidence_threshold}")
        print(f"   Stability threshold: {self.stability_threshold}")
    
    def extract_features_improved(self, lm_list, img_shape):
        """Extract features with improved normalization"""
        if len(lm_list) == 0:
            return None
        
        try:
            features = []
            h, w, _ = img_shape
            
            # Extract normalized coordinates
            for landmark in lm_list:
                # landmark format: [id, x, y]
                x_norm = landmark[1] / w  # Normalize x by width
                y_norm = landmark[2] / h  # Normalize y by height
                features.extend([x_norm, y_norm])
            
            # Ensure we have exactly 42 features (21 landmarks * 2 coordinates)
            while len(features) < 42:
                features.extend([0.0, 0.0])
            
            # Take only first 42 features
            features = features[:42]
            
            # Convert to numpy array
            features_array = np.array(features, dtype=np.float32)
            
            # Check for invalid values
            if np.isnan(features_array).any() or np.isinf(features_array).any():
                print("⚠️ Invalid features detected (NaN/Inf)")
                return None
            
            # Apply feature scaling if available
            if self.scaler is not None:
                features_array = self.scaler.transform(features_array.reshape(1, -1))[0]
            
            if self.debug_mode:
                print(f"Features extracted: {features_array.shape}, Range: [{features_array.min():.3f}, {features_array.max():.3f}]")
            
            return features_array
            
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return None
    
    def predict_gesture(self, features):
        """Predict gesture with improved confidence handling"""
        try:
            # Make prediction
            prediction = self.model.predict(features.reshape(1, -1), verbose=0)[0]
            
            # Get predicted class and confidence
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]
            
            # Get top 3 predictions for debugging
            top_3_indices = np.argsort(prediction)[-3:][::-1]
            top_3_confidences = prediction[top_3_indices]
            top_3_gestures = [self.label_encoder.classes_[i] for i in top_3_indices]
            
            # Get main prediction
            gesture = self.label_encoder.classes_[predicted_class]
            
            if self.debug_mode:
                print(f"Top 3 predictions: {list(zip(top_3_gestures, top_3_confidences))}")
            
            return gesture, confidence, top_3_gestures, top_3_confidences
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return None, 0.0, [], []
    
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

        def _get_font_path_for_language(self):
            base = os.path.join(os.path.dirname(__file__), "fonts")
            mapping = {
                'English': os.path.join(base, "NotoSans-Regular.ttf"),
                'Hindi':   os.path.join(base, "NotoSansDevanagari-Regular.ttf"),
                'Marathi': os.path.join(base, "NotoSansDevanagari-Regular.ttf"),
                'Tamil':   os.path.join(base, "NotoSansTamil-Regular.ttf"),
                'Telugu':  os.path.join(base, "NotoSansTelugu-Regular.ttf"),
                'Bengali': os.path.join(base, "NotoSansBengali-Regular.ttf"),
                'Gujarati':os.path.join(base, "NotoSansGujarati-Regular.ttf"),
                'Kannada': os.path.join(base, "NotoSansKannada-Regular.ttf"),
                'Malayalam':os.path.join(base, "NotoSansMalayalam-Regular.ttf"),
                'Punjabi': os.path.join(base, "NotoSansGurmukhi-Regular.ttf"),
            }
            # fallback to a general font if language not found
            return mapping.get(self.current_language, mapping['English'])

        def put_unicode_text(self, img, text, pos, font_size=28, color=(255,255,255)):
            """Draw Unicode text onto an OpenCV BGR image using PIL.
            pos is (x, y) in pixels (top-left of text)."""
            if not text:
                return img
            try:
                # Convert BGR->RGB and to PIL
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)

                font_path = self._get_font_path_for_language()
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                else:
                    # If font file missing, fall back to default PIL font
                    font = ImageFont.load_default()
                    print(f"⚠️ Font not found: {font_path}. Using default font (may not support Unicode).")

                # Pillow expects color as RGB tuple
                draw.text((int(pos[0]), int(pos[1])), str(text), font=font, fill=(int(color[0]), int(color[1]), int(color[2])))

                # Convert back to OpenCV BGR
                return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"❌ Unicode render error: {e}")
                return img


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
    
    def draw_landmarks_enhanced(self, img, lm_list):
        """Draw enhanced hand landmarks"""
        if len(lm_list) == 0:
            return img
        
        # Define hand connections (MediaPipe hand model)
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        # Draw connections
        for connection in connections:
            if connection[0] < len(lm_list) and connection[1] < len(lm_list):
                x1, y1 = lm_list[connection[0]][1], lm_list[connection[0]][2]
                x2, y2 = lm_list[connection[1]][1], lm_list[connection[1]][2]
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw landmarks
        for lm in lm_list:
            cv2.circle(img, (lm[1], lm[2]), 5, (255, 0, 255), -1)
        
        return img
    
    def run_real_time_improved(self, camera_index=0):
        """Run improved real-time ASL recognition"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_index}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("🚀 Improved Real-time ASL Recognition Started!")
        print("\n📋 Controls:")
        print("  SPACE    - Add predicted letter to text")
        print("  BACKSPACE- Delete last letter")
        print("  ENTER    - Translate text")
        print("  'c'      - Clear all text")
        print("  'l'      - Change language")
        print("  'd'      - Toggle debug mode")
        print("  '+'/'-'  - Adjust confidence threshold")
        print("  'q'      - Quit")
        print("  's'      - Save current sentence")
        print("\n🎯 Recognition Settings:")
        print(f"  Confidence threshold: {self.confidence_threshold}")
        print(f"  Stability requirement: {self.stability_threshold} consistent frames")
        
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
            
            # Find hands with improved detection
            img = self.detector.find_hands(img, draw=False)  # We'll draw our own landmarks
            lm_list = self.detector.find_position(img, draw=False)
            
            # Initialize prediction variables
            current_prediction = "No hand detected"
            confidence = 0.0
            is_stable = False
            top_3_predictions = []
            
            if len(lm_list) > 0:
                # Draw enhanced landmarks
                img = self.draw_landmarks_enhanced(img, lm_list)
                
                # Extract features
                features = self.extract_features_improved(lm_list, img.shape)
                
                if features is not None:
                    # Make prediction
                    gesture, conf, top_3_gestures, top_3_conf = self.predict_gesture(features)
                    
                    if gesture is not None:
                        # Stabilize prediction
                        stable_gesture, is_stable = self.stabilize_prediction(gesture, conf)
                        
                        current_prediction = stable_gesture if stable_gesture else f"{gesture} (unstable)"
                        confidence = conf
                        top_3_predictions = list(zip(top_3_gestures, top_3_conf))
                else:
                    current_prediction = "Feature extraction failed"
            
            # Calculate FPS
            end_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (end_time - start_time)
            fps_counter.append(fps)
            avg_fps = np.mean(fps_counter)
            
            # Draw merged UI (Fixed UI style + top-3 debug)
            img = self.draw_enhanced_ui(img, current_prediction, confidence, is_stable, 
                                        top_3_predictions, avg_fps, frame_count)
            
            cv2.imshow('Improved ASL Recognition', img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space - add letter
                if current_prediction not in ["No hand detected", "Feature extraction failed"] and is_stable:
                    # Remove "(unstable)" suffix if present
                    clean_prediction = current_prediction.split(" (")[0]
                    self.recognized_text += clean_prediction
                    print(f"✅ Added: {clean_prediction} -> Text: '{self.recognized_text}'")
                    
                    # Reset stability tracking after successful addition
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

                    # Create popup image
                    popup_w, popup_h = 800, 200
                    popup_img = np.zeros((popup_h, popup_w, 3), dtype=np.uint8)

                    # Header
                    cv2.putText(popup_img, f"Translated to {self.current_language}:",
                                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    # Wrapped Unicode-safe lines
                    wrapped_lines = self.wrap_text(self.translated_text, 40)
                    base_y = 90
                    for i, line in enumerate(wrapped_lines[:3]):  # Max 3 lines
                        popup_img = self.put_unicode_text(popup_img, line,
                                                        (20, base_y + i * 40),
                                                        font_size=30, color=(255, 255, 255))

                    cv2.imshow("Translation Result", popup_img)
                    cv2.waitKey(2000)  # Display for 2 seconds
                    cv2.destroyWindow("Translation Result")


                '''
            elif key == 13:  # Enter - translate
                if self.recognized_text:
                    self.translated_text = self.translate_text(self.recognized_text)
                    print(f"🌐 Translated to {self.current_language}: '{self.translated_text}'")
                '''
                    
            elif key == ord('c'):  # Clear
                self.recognized_text = ""
                self.translated_text = ""
                self.stable_prediction = None
                self.stable_count = 0
                print("🧹 Text cleared")
            
            elif key == ord('l'):  # Change language
                self.change_language()
                continue
            
            elif key == ord('d'):  # Toggle debug mode
                self.debug_mode = not self.debug_mode
                print(f"🐛 Debug mode: {'ON' if self.debug_mode else 'OFF'}")
            
            elif key == ord('s'):  # Save sentence
                if self.recognized_text:
                    self.save_sentence()
            
            elif key == ord('+') or key == ord('='):
                self.confidence_threshold = min(0.99, self.confidence_threshold + 0.05)
                print(f"🎯 Confidence threshold increased to: {self.confidence_threshold:.2f}")
            
            elif key == ord('-'):
                self.confidence_threshold = max(0.50, self.confidence_threshold - 0.05)
                print(f"🎯 Confidence threshold decreased to: {self.confidence_threshold:.2f}")
            
            elif key == ord('q'):  # Quit
                print("👋 Goodbye!")
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def draw_enhanced_ui(self, img, prediction, confidence, is_stable, top_3, fps, frame_count):
        """Draw merged user interface (Fixed UI style + improved debug/top3)"""
        h, w, _ = img.shape
        
        # Create a more opaque overlay for better visibility (Fixed UI style)
        overlay = img.copy()
        # Top panel
        cv2.rectangle(overlay, (10, 10), (w-10, 160), (0, 0, 0), -1)
        # Bottom panel
        cv2.rectangle(overlay, (10, h-200), (w-10, h-10), (0, 0, 0), -1)
        
        # Blend with higher opacity
        img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        
        # Add white borders for better visibility
        cv2.rectangle(img, (10, 10), (w-10, 160), (255, 255, 255), 2)
        cv2.rectangle(img, (10, h-200), (w-10, h-10), (255, 255, 255), 2)
        
        # Current prediction with larger font and bright colors
        pred_color = (0, 255, 0) if is_stable else (0, 165, 255)  # Green if stable, orange if not
        cv2.putText(img, f'Prediction: {prediction}', (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, pred_color, 2)
        
        # Confidence with color coding
        conf_color = (0, 255, 0) if confidence > self.confidence_threshold else (0, 0, 255)
        cv2.putText(img, f'Confidence: {confidence:.3f}', (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, conf_color, 2)
        
        # Stability indicator
        stability_text = f'Stable: {"YES" if is_stable else "NO"} ({self.stable_count}/{self.stability_threshold})'
        stability_color = (0, 255, 0) if is_stable else (0, 255, 255)
        cv2.putText(img, stability_text, (20, 115), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, stability_color, 2)
        
        # Language and threshold / FPS
        cv2.putText(img, f'Language: {self.current_language} | Threshold: {self.confidence_threshold:.2f} | FPS: {fps:.1f}', 
                   (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Debug info - show top 3 predictions (from Improved version)
        if self.debug_mode and top_3:
            y_offset = 170
            cv2.putText(img, 'Top 3 Predictions:', (w-340, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            for i, (gesture, conf) in enumerate(top_3[:3]):
                y_offset += 25
                cv2.putText(img, f'{i+1}. {gesture}: {conf:.3f}', (w-340, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Recognized text (with wrapping for long text)
        text_lines = self.wrap_text(f'Text: {self.recognized_text}', 60)
        for i, line in enumerate(text_lines[:2]):  # Show max 2 lines
            cv2.putText(img, line, (20, h-170 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
                # Translated text (Unicode safe)
            if self.translated_text:
                # Wrap text to avoid overly long lines
                trans_lines = self.wrap_text(f'Translated: {self.translated_text}', 50)
                base_y = h - 120
                for i, line in enumerate(trans_lines[:2]):  # only show first 2 lines
                    y = base_y + i * 28
                    img = self.put_unicode_text(img, line, (20, y), font_size=26, color=(0, 255, 255))

        # Enhanced controls with better visibility (includes +/-)
        controls = [
            'SPACE: Add Letter | ENTER: Translate | C: Clear | D: Debug | S: Save | Q: Quit',
            'BACKSPACE: Delete | L: Language | +/-: Adjust Threshold'
        ]
        cv2.putText(img, controls[0], (20, h-60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, controls[1], (20, h-35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # FPS counter in top right (redundant with top text but handy)
        if fps > 0:
            cv2.putText(img, f'FPS: {fps:.1f}', (w-120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return img
    
    def wrap_text(self, text, width):
        """Wrap text to fit in specified width"""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            if len(' '.join(current_line + [word])) <= width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def change_language(self):
        """Non-blocking language selection inside OpenCV window (Windows/Linux compatible)"""
        languages = list(self.target_languages.keys())
        selected_index = languages.index(self.current_language)

        while True:
            # Create menu window
            menu_img = np.zeros((400, 600, 3), dtype=np.uint8)
            cv2.putText(menu_img, "Select Language", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            for i, lang in enumerate(languages):
                color = (0, 255, 0) if i == selected_index else (255, 255, 255)
                cv2.putText(menu_img, f"{i+1}. {lang}", (40, 80 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.putText(menu_img, "↑/↓: Move  Enter: Select  Q/Esc: Cancel",
                        (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Language Menu", menu_img)

            key = cv2.waitKey(0) & 0xFFFFFFFF

            # Cancel
            if key in [27, ord('q')]:  
                cv2.destroyWindow("Language Menu")
                return

            # Up arrow (Windows/Linux) or W
            elif key in [2490368, 82, ord('w')]:
                selected_index = (selected_index - 1) % len(languages)

            # Down arrow (Windows/Linux) or S
            elif key in [2621440, 84, ord('s')]:
                selected_index = (selected_index + 1) % len(languages)

            # Enter = Select
            elif key == 13:
                self.current_language = languages[selected_index]
                print(f"✅ Language changed to: {self.current_language}")
                if self.recognized_text:
                    self.translated_text = self.translate_text(self.recognized_text)
                    print(f"🔄 Re-translated: '{self.translated_text}'")
                cv2.destroyWindow("Language Menu")
                return


    '''
    def change_language(self):
        """Change target language (Fixed UI simple menu)"""
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
    
    def save_sentence(self):
        """Save the current sentence to a file"""
        if not self.recognized_text:
            print("⚠️ No text to save!")
            return
        
        try:
            filename = "asl_sentences.txt"
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(f"Original: {self.recognized_text}\n")
                if self.translated_text:
                    f.write(f"Translated ({self.current_language}): {self.translated_text}\n")
                f.write(f"Timestamp: {cv2.getTickCount()}\n")
                f.write("-" * 50 + "\n")
            
            print(f"💾 Sentence saved to '{filename}'")
        except Exception as e:
            print(f"❌ Error saving sentence: {e}")
    '''


def main():
    """Main function with improved error handling"""
    print("🤖 Improved ASL Real-time Recognition System")
    print("=" * 50)
    
    # Check for required files with specific guidance
    required_files = {
        "improved_asl_model.h5": "Run the improved model trainer first!",
        "improved_label_encoder.pkl": "Label encoder missing - retrain the model!",
        "feature_scaler.pkl": "Feature scaler missing - retrain the model!",
        "hand_detection.py": "Hand detection module missing!"
    }
    
    missing_files = []
    for file, message in required_files.items():
        if not os.path.exists(file):
            missing_files.append((file, message))
    
    if missing_files:
        print("❌ Missing required files:")
        for file, message in missing_files:
            print(f"   - {file}: {message}")
        return
    
    # Initialize predictor
    try:
        predictor = ImprovedASLPredictor()
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        return
    
    # Get camera index
    try:
        camera_input = input("Camera index (0 for default, 1, 2, etc.): ").strip()
        camera_index = int(camera_input) if camera_input else 0
    except ValueError:
        print("Using default camera (index 0)")
        camera_index = 0
    
    # Start improved real-time recognition
    predictor.run_real_time_improved(camera_index)

if __name__ == "__main__":
    main()
