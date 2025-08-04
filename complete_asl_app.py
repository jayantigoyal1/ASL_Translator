<<<<<<< HEAD
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image, ImageTk
from googletrans import Translator
from collections import deque
import threading
import time
import os
from hand_detection import HandDetector

class ASLTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language to Text Translator")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2c3e50')
        
        # Initialize components
        self.detector = HandDetector()
        self.translator = Translator()
        self.model = None
        self.label_encoder = None
        self.scaler = None
        
        # Camera variables
        self.cap = None
        self.camera_running = False
        self.current_prediction = "No hand"
        self.current_confidence = 0.0
        
        # Improved prediction system from real-time predictor
        self.prediction_buffer = deque(maxlen=5)  # Reduced buffer for faster response
        self.confidence_threshold = 0.8  # Higher threshold for better accuracy
        self.min_detection_confidence = 0.5
        
        # Stability tracking (from working predictor)
        self.stable_prediction = None
        self.stable_count = 0
        self.stability_threshold = 3  # Need 3 consistent predictions
        
        # Text variables
        self.recognized_text = ""
        self.translated_text = ""
        
        # Languages
        self.languages = {
            'English': 'en', 'Hindi': 'hi', 'Tamil': 'ta', 'Telugu': 'te',
            'Bengali': 'bn', 'Marathi': 'mr', 'Gujarati': 'gu', 'Kannada': 'kn'
        }
        
        self.load_model()
        self.create_gui()
        
    def load_model(self):
        """Load the improved trained model with fallback system"""
        # Try to load models in order of preference (from real-time predictor)
        model_files = [
            ("improved_asl_model.h5", "improved_label_encoder.pkl", "feature_scaler.pkl"),
            ("best_model.h5", "label_encoder.pkl", None),
            ("asl_model.h5", "label_encoder.pkl", None)
        ]
        
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
                        self.scaler = None
                    
                    return  # Success, exit the loop
                    
                except Exception as e:
                    print(f"❌ Error loading {model_path}: {e}")
                    continue
        
        # If we get here, no model was loaded successfully
        messagebox.showerror("Error", "No model found!\nTried:\n- improved_asl_model.h5\n- best_model.h5\n- asl_model.h5\n\nPlease ensure model files exist.")
            
    def create_gui(self):
        """Create the main GUI"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="Sign Language to Text", 
                              font=('Arial', 28, 'bold'), bg='#2c3e50', fg='white')
        title_label.pack(expand=True)
        
        # Main content
        main_frame = tk.Frame(self.root, bg='#34495e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left side - Camera
        left_frame = tk.Frame(main_frame, bg='#34495e')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Camera display
        self.camera_label = tk.Label(left_frame, bg='black', text="Camera Off\n\nClick Start Camera", 
                                   fg='white', font=('Arial', 18))
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=(0, 20))
        
        # Camera controls
        cam_control_frame = tk.Frame(left_frame, bg='#34495e')
        cam_control_frame.pack(fill=tk.X, pady=(10, 0), padx=(0, 20))
        
        self.start_btn = tk.Button(cam_control_frame, text="Start Camera", 
                                 command=self.start_camera, bg='#27ae60', fg='white',
                                 font=('Arial', 12, 'bold'), relief=tk.FLAT, padx=20)
        self.start_btn.pack(side=tk.LEFT, pady=5)
        
        self.stop_btn = tk.Button(cam_control_frame, text="Stop Camera", 
                                command=self.stop_camera, bg='#e74c3c', fg='white',
                                font=('Arial', 12, 'bold'), relief=tk.FLAT, padx=20)
        self.stop_btn.pack(side=tk.LEFT, padx=(10, 0), pady=5)
        
        # Right side - Controls and text
        right_frame = tk.Frame(main_frame, bg='#34495e', width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)
        
        # Current prediction
        pred_frame = tk.LabelFrame(right_frame, text="Current Prediction", 
                                 bg='#34495e', fg='white', font=('Arial', 14, 'bold'))
        pred_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.pred_label = tk.Label(pred_frame, text="--", font=('Arial', 36, 'bold'), 
                                 bg='#34495e', fg='#3498db')
        self.pred_label.pack(pady=20)
        
        self.conf_label = tk.Label(pred_frame, text="Confidence: 0%", 
                                 font=('Arial', 12), bg='#34495e', fg='#bdc3c7')
        self.conf_label.pack()
        
        # Stability indicator (new from real-time predictor)
        self.stability_label = tk.Label(pred_frame, text="Stability: 0/3", 
                                      font=('Arial', 10), bg='#34495e', fg='#95a5a6')
        self.stability_label.pack()
        
        # Action buttons
        btn_frame = tk.Frame(right_frame, bg='#34495e')
        btn_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.add_btn = tk.Button(btn_frame, text="Add Letter", command=self.add_letter,
                               bg='#3498db', fg='white', font=('Arial', 14, 'bold'),
                               relief=tk.FLAT, pady=10)
        self.add_btn.pack(fill=tk.X, pady=2)
        
        self.space_btn = tk.Button(btn_frame, text="Add Space", command=self.add_space,
                                 bg='#f39c12', fg='white', font=('Arial', 14, 'bold'),
                                 relief=tk.FLAT, pady=10)
        self.space_btn.pack(fill=tk.X, pady=2)
        
        self.delete_btn = tk.Button(btn_frame, text="Delete Last", command=self.delete_last,
                                  bg='#95a5a6', fg='white', font=('Arial', 14, 'bold'),
                                  relief=tk.FLAT, pady=10)
        self.delete_btn.pack(fill=tk.X, pady=2)
        
        # Confidence threshold control (new feature)
        threshold_frame = tk.LabelFrame(right_frame, text="Confidence Threshold", 
                                      bg='#34495e', fg='white', font=('Arial', 12, 'bold'))
        threshold_frame.pack(fill=tk.X, pady=(0, 10))
        
        threshold_control_frame = tk.Frame(threshold_frame, bg='#34495e')
        threshold_control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.threshold_var = tk.DoubleVar(value=self.confidence_threshold)
        threshold_scale = tk.Scale(threshold_control_frame, from_=0.5, to=0.95, resolution=0.05,
                                 orient=tk.HORIZONTAL, variable=self.threshold_var,
                                 bg='#34495e', fg='white', highlightbackground='#34495e',
                                 command=self.update_threshold)
        threshold_scale.pack(fill=tk.X)
        
        self.threshold_label = tk.Label(threshold_frame, text=f"Current: {self.confidence_threshold}", 
                                      font=('Arial', 10), bg='#34495e', fg='#bdc3c7')
        self.threshold_label.pack()
        
        # Language selection
        lang_frame = tk.LabelFrame(right_frame, text="Target Language", 
                                 bg='#34495e', fg='white', font=('Arial', 14, 'bold'))
        lang_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.lang_var = tk.StringVar(value="English")
        lang_combo = ttk.Combobox(lang_frame, textvariable=self.lang_var, 
                                values=list(self.languages.keys()), state="readonly",
                                font=('Arial', 12))
        lang_combo.pack(fill=tk.X, padx=10, pady=10)
        
        # Recognized Text area
        text_frame = tk.LabelFrame(right_frame, text="Recognized Text", 
                                 bg='#34495e', fg='white', font=('Arial', 14, 'bold'))
        text_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Text frame with scrollbar
        text_scroll_frame = tk.Frame(text_frame, bg='#34495e')
        text_scroll_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        self.text_display = tk.Text(text_scroll_frame, height=4, font=('Arial', 12), 
                                  bg='#ecf0f1', fg='#2c3e50', wrap=tk.WORD)
        text_scrollbar = tk.Scrollbar(text_scroll_frame, orient=tk.VERTICAL, command=self.text_display.yview)
        self.text_display.configure(yscrollcommand=text_scrollbar.set)
        
        self.text_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Translation area (separate frame)
        trans_frame = tk.LabelFrame(right_frame, text="Translated Text", 
                                  bg='#34495e', fg='white', font=('Arial', 14, 'bold'))
        trans_frame.pack(fill=tk.BOTH, expand=True)
        
        # Translation frame with scrollbar
        trans_scroll_frame = tk.Frame(trans_frame, bg='#34495e')
        trans_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))
        
        self.trans_display = tk.Text(trans_scroll_frame, height=4, font=('Arial', 12), 
                                   bg='#d5dbdb', fg='#2c3e50', wrap=tk.WORD)
        trans_scrollbar = tk.Scrollbar(trans_scroll_frame, orient=tk.VERTICAL, command=self.trans_display.yview)
        self.trans_display.configure(yscrollcommand=trans_scrollbar.set)
        
        self.trans_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        trans_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bottom buttons
        bottom_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        bottom_frame.pack(fill=tk.X)
        bottom_frame.pack_propagate(False)
        
        self.translate_btn = tk.Button(bottom_frame, text="Translate Text", 
                                     command=self.translate_text, bg='#9b59b6', 
                                     fg='white', font=('Arial', 16, 'bold'),
                                     relief=tk.FLAT, padx=30)
        self.translate_btn.pack(side=tk.LEFT, padx=20, pady=15)
        
        self.clear_btn = tk.Button(bottom_frame, text="Clear All", 
                                 command=self.clear_all, bg='#e67e22', 
                                 fg='white', font=('Arial', 16, 'bold'),
                                 relief=tk.FLAT, padx=30)
        self.clear_btn.pack(side=tk.LEFT, padx=10, pady=15)
        
        self.quit_btn = tk.Button(bottom_frame, text="Quit", 
                                command=self.quit_app, bg='#c0392b', 
                                fg='white', font=('Arial', 16, 'bold'),
                                relief=tk.FLAT, padx=30)
        self.quit_btn.pack(side=tk.RIGHT, padx=20, pady=15)
        
    def update_threshold(self, value):
        """Update confidence threshold"""
        self.confidence_threshold = float(value)
        self.threshold_label.configure(text=f"Current: {self.confidence_threshold:.2f}")
        
    def start_camera(self):
        """Start camera feed"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded!")
            return
            
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera!")
            return
            
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
            
        self.camera_running = True
        self.update_camera()
        
    def stop_camera(self):
        """Stop camera feed"""
        self.camera_running = False
        if self.cap:
            self.cap.release()
        self.camera_label.configure(image="", text="Camera Off\n\nClick Start Camera")
        # Reset prediction state
        self.stable_prediction = None
        self.stable_count = 0
        self.prediction_buffer.clear()
        
    def extract_features_improved(self, lm_list, img_shape):
        """Improved feature extraction from real-time predictor"""
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
            
            return features_array
            
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return None
    
    def predict_gesture_improved(self, features):
        """Improved gesture prediction from real-time predictor"""
        try:
            # Make prediction
            prediction = self.model.predict(features.reshape(1, -1), verbose=0)[0]
            
            # Get predicted class and confidence
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]
            
            # Get gesture
            gesture = self.label_encoder.classes_[predicted_class]
            
            return gesture, confidence
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return None, 0.0
    
    def stabilize_prediction(self, gesture, confidence):
        """Stabilize predictions for consistent recognition (from real-time predictor)"""
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
        
    def update_camera(self):
        """Update camera feed and predictions with improved logic"""
        if not self.camera_running:
            return
            
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            # Hand detection and prediction
            frame = self.detector.find_hands(frame, draw=True)
            landmarks = self.detector.find_position(frame, draw=False)
            
            # Initialize prediction variables
            current_prediction = "No hand detected"
            confidence = 0.0
            is_stable = False
            
            if len(landmarks) > 0:
                # Extract features using improved method
                features = self.extract_features_improved(landmarks, frame.shape)
                
                if features is not None:
                    # Make prediction using improved method
                    gesture, conf = self.predict_gesture_improved(features)
                    
                    if gesture is not None:
                        # Stabilize prediction
                        stable_gesture, is_stable = self.stabilize_prediction(gesture, conf)
                        
                        current_prediction = stable_gesture if stable_gesture else f"{gesture} (unstable)"
                        confidence = conf
                else:
                    current_prediction = "Feature extraction failed"
            else:
                # Clear prediction state when no hand detected
                self.stable_prediction = None
                self.stable_count = 0
            
            # Update prediction display
            self.current_prediction = current_prediction
            self.current_confidence = confidence
            
            # Update UI elements
            self.pred_label.configure(text=current_prediction)
            self.conf_label.configure(text=f"Confidence: {confidence:.1%}")
            
            # Update stability indicator
            stability_color = '#27ae60' if is_stable else '#95a5a6'
            stability_text = f"Stability: {self.stable_count}/{self.stability_threshold}"
            self.stability_label.configure(text=stability_text, fg=stability_color)
            
            # Convert frame for tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (500, 375))
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            
            self.camera_label.configure(image=photo, text="")
            self.camera_label.image = photo
            
        # Schedule next update
        self.root.after(50, self.update_camera)
    
    def add_letter(self):
        """Add current prediction to text with improved logic"""
        if (self.current_prediction not in ["No hand detected", "Feature extraction failed", "--"] 
            and self.stable_count >= self.stability_threshold 
            and self.current_confidence > self.confidence_threshold):
            
            # Clean prediction (remove unstable suffix)
            clean_prediction = self.current_prediction.split(" (")[0]
            self.recognized_text += clean_prediction
            self.update_text_display()
            
            # Reset stability tracking after successful addition
            self.stable_prediction = None
            self.stable_count = 0
            self.prediction_buffer.clear()
            
            print(f"✅ Added: {clean_prediction} -> Text: '{self.recognized_text}'")
        else:
            print(f"⚠️ Cannot add: prediction='{self.current_prediction}', stable={self.stable_count >= self.stability_threshold}, conf={self.current_confidence:.3f}")
    
    def add_space(self):
        """Add space to text"""
        self.recognized_text += " "
        self.update_text_display()
    
    def delete_last(self):
        """Delete last character"""
        if self.recognized_text:
            self.recognized_text = self.recognized_text[:-1]
            self.update_text_display()
    
    def update_text_display(self):
        """Update text display with auto-scroll"""
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(1.0, self.recognized_text)
        # Auto-scroll to show the latest text
        self.text_display.see(tk.END)
    
    def translate_text(self):
        """Translate text to selected language with better error handling"""
        if not self.recognized_text.strip():
            messagebox.showwarning("Warning", "No text to translate!")
            return
            
        target_lang = self.languages[self.lang_var.get()]
        
        if target_lang == 'en':
            self.translated_text = self.recognized_text
        else:
            try:
                print(f"🌐 Translating '{self.recognized_text}' to {self.lang_var.get()}...")
                result = self.translator.translate(self.recognized_text, dest=target_lang)
                self.translated_text = result.text
                print(f"✅ Translation successful: '{self.translated_text}'")
            except Exception as e:
                print(f"❌ Translation error: {e}")
                self.translated_text = f"Translation error: {str(e)}"
                messagebox.showerror("Translation Error", f"Failed to translate text:\n{str(e)}")
        
        # Update translation display
        self.trans_display.delete(1.0, tk.END)
        self.trans_display.insert(1.0, self.translated_text)
        
        # Auto-scroll to show the text
        self.trans_display.see(tk.END)
    
    def clear_all(self):
        """Clear all text and reset prediction state"""
        self.recognized_text = ""
        self.translated_text = ""
        self.text_display.delete(1.0, tk.END)
        self.trans_display.delete(1.0, tk.END)
        
        # Reset prediction state
        self.stable_prediction = None
        self.stable_count = 0
        self.prediction_buffer.clear()
    
    def quit_app(self):
        """Quit application"""
        self.stop_camera()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = ASLTranslatorApp(root)
    root.mainloop()

if __name__ == "__main__":
=======
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image, ImageTk
from googletrans import Translator
from collections import deque
import threading
import time
from hand_detection import HandDetector

class ASLTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language to Text Translator")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2c3e50')
        
        # Initialize components
        self.detector = HandDetector()
        self.translator = Translator()
        self.model = None
        self.label_encoder = None
        
        # Camera variables
        self.cap = None
        self.camera_running = False
        self.current_prediction = "No hand"
        self.current_confidence = 0.0
        self.prediction_buffer = deque(maxlen=8)
        
        # Text variables
        self.recognized_text = ""
        self.translated_text = ""
        
        # Languages
        self.languages = {
            'English': 'en', 'Hindi': 'hi', 'Tamil': 'ta', 'Telugu': 'te',
            'Bengali': 'bn', 'Marathi': 'mr', 'Gujarati': 'gu', 'Kannada': 'kn'
        }
        
        self.load_model()
        self.create_gui()
        
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model("asl_model.h5")
            with open("label_encoder.pkl", 'rb') as f:
                self.label_encoder = pickle.load(f)
            print("✅ Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Model not found!\nPlease train the model first:\npython model_trainer.py")
            
    def create_gui(self):
        """Create the main GUI"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="Sign Language to Text", 
                              font=('Arial', 28, 'bold'), bg='#2c3e50', fg='white')
        title_label.pack(expand=True)
        
        # Main content
        main_frame = tk.Frame(self.root, bg='#34495e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left side - Camera
        left_frame = tk.Frame(main_frame, bg='#34495e')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Camera display
        self.camera_label = tk.Label(left_frame, bg='black', text="Camera Off\n\nClick Start Camera", 
                                   fg='white', font=('Arial', 18))
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=(0, 20))
        
        # Camera controls
        cam_control_frame = tk.Frame(left_frame, bg='#34495e')
        cam_control_frame.pack(fill=tk.X, pady=(10, 0), padx=(0, 20))
        
        self.start_btn = tk.Button(cam_control_frame, text="Start Camera", 
                                 command=self.start_camera, bg='#27ae60', fg='white',
                                 font=('Arial', 12, 'bold'), relief=tk.FLAT, padx=20)
        self.start_btn.pack(side=tk.LEFT, pady=5)
        
        self.stop_btn = tk.Button(cam_control_frame, text="Stop Camera", 
                                command=self.stop_camera, bg='#e74c3c', fg='white',
                                font=('Arial', 12, 'bold'), relief=tk.FLAT, padx=20)
        self.stop_btn.pack(side=tk.LEFT, padx=(10, 0), pady=5)
        
        # Right side - Controls and text
        right_frame = tk.Frame(main_frame, bg='#34495e', width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)
        
        # Current prediction
        pred_frame = tk.LabelFrame(right_frame, text="Current Prediction", 
                                 bg='#34495e', fg='white', font=('Arial', 14, 'bold'))
        pred_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.pred_label = tk.Label(pred_frame, text="--", font=('Arial', 36, 'bold'), 
                                 bg='#34495e', fg='#3498db')
        self.pred_label.pack(pady=20)
        
        self.conf_label = tk.Label(pred_frame, text="Confidence: 0%", 
                                 font=('Arial', 12), bg='#34495e', fg='#bdc3c7')
        self.conf_label.pack()
        
        # Action buttons
        btn_frame = tk.Frame(right_frame, bg='#34495e')
        btn_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.add_btn = tk.Button(btn_frame, text="Add Letter", command=self.add_letter,
                               bg='#3498db', fg='white', font=('Arial', 14, 'bold'),
                               relief=tk.FLAT, pady=10)
        self.add_btn.pack(fill=tk.X, pady=2)
        
        self.space_btn = tk.Button(btn_frame, text="Add Space", command=self.add_space,
                                 bg='#f39c12', fg='white', font=('Arial', 14, 'bold'),
                                 relief=tk.FLAT, pady=10)
        self.space_btn.pack(fill=tk.X, pady=2)
        
        self.delete_btn = tk.Button(btn_frame, text="Delete Last", command=self.delete_last,
                                  bg='#95a5a6', fg='white', font=('Arial', 14, 'bold'),
                                  relief=tk.FLAT, pady=10)
        self.delete_btn.pack(fill=tk.X, pady=2)
        
        # Language selection
        lang_frame = tk.LabelFrame(right_frame, text="Target Language", 
                                 bg='#34495e', fg='white', font=('Arial', 14, 'bold'))
        lang_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.lang_var = tk.StringVar(value="English")
        lang_combo = ttk.Combobox(lang_frame, textvariable=self.lang_var, 
                                values=list(self.languages.keys()), state="readonly",
                                font=('Arial', 12))
        lang_combo.pack(fill=tk.X, padx=10, pady=10)
        
        # Text areas
        text_frame = tk.LabelFrame(right_frame, text="Recognized Text", 
                                 bg='#34495e', fg='white', font=('Arial', 14, 'bold'))
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.text_display = tk.Text(text_frame, height=6, font=('Arial', 14), 
                                  bg='#ecf0f1', fg='#2c3e50', wrap=tk.WORD)
        self.text_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))
        
        # Translation area
        self.trans_display = tk.Text(text_frame, height=4, font=('Arial', 12), 
                                   bg='#d5dbdb', fg='#2c3e50', wrap=tk.WORD)
        self.trans_display.pack(fill=tk.X, padx=10, pady=(5, 10))
        
        # Bottom buttons
        bottom_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        bottom_frame.pack(fill=tk.X)
        bottom_frame.pack_propagate(False)
        
        self.translate_btn = tk.Button(bottom_frame, text="Translate Text", 
                                     command=self.translate_text, bg='#9b59b6', 
                                     fg='white', font=('Arial', 16, 'bold'),
                                     relief=tk.FLAT, padx=30)
        self.translate_btn.pack(side=tk.LEFT, padx=20, pady=15)
        
        self.clear_btn = tk.Button(bottom_frame, text="Clear All", 
                                 command=self.clear_all, bg='#e67e22', 
                                 fg='white', font=('Arial', 16, 'bold'),
                                 relief=tk.FLAT, padx=30)
        self.clear_btn.pack(side=tk.LEFT, padx=10, pady=15)
        
        self.quit_btn = tk.Button(bottom_frame, text="Quit", 
                                command=self.quit_app, bg='#c0392b', 
                                fg='white', font=('Arial', 16, 'bold'),
                                relief=tk.FLAT, padx=30)
        self.quit_btn.pack(side=tk.RIGHT, padx=20, pady=15)
        
    def start_camera(self):
        """Start camera feed"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded!")
            return
            
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera!")
            return
            
        self.camera_running = True
        self.update_camera()
        
    def stop_camera(self):
        """Stop camera feed"""
        self.camera_running = False
        if self.cap:
            self.cap.release()
        self.camera_label.configure(image="", text="Camera Off\n\nClick Start Camera")
        
    def update_camera(self):
        """Update camera feed and predictions"""
        if not self.camera_running:
            return
            
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            # Hand detection and prediction
            frame = self.detector.find_hands(frame, draw=True)
            landmarks = self.detector.find_position(frame, draw=False)
            
            if len(landmarks) > 0:
                # Extract features and predict
                features = self.extract_features(landmarks, frame.shape)
                prediction = self.model.predict(features.reshape(1, -1), verbose=0)[0]
                
                predicted_class = np.argmax(prediction)
                confidence = prediction[predicted_class]
                gesture = self.label_encoder.classes_[predicted_class]
                
                # Smooth predictions
                if confidence > 0.8:
                    self.prediction_buffer.append(gesture)
                
                if len(self.prediction_buffer) > 0:
                    unique, counts = np.unique(self.prediction_buffer, return_counts=True)
                    self.current_prediction = unique[np.argmax(counts)]
                    self.current_confidence = confidence
                else:
                    self.current_prediction = "Low confidence"
                    self.current_confidence = confidence
            else:
                self.current_prediction = "No hand"
                self.current_confidence = 0.0
            
            # Update prediction display
            self.pred_label.configure(text=self.current_prediction)
            self.conf_label.configure(text=f"Confidence: {self.current_confidence:.1%}")
            
            # Convert frame for tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (500, 375))
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            
            self.camera_label.configure(image=photo, text="")
            self.camera_label.image = photo
            
        # Schedule next update
        self.root.after(50, self.update_camera)
        
    def extract_features(self, landmarks, img_shape):
        """Extract features from landmarks"""
        features = []
        h, w, _ = img_shape
        
        for landmark in landmarks:
            x_norm = landmark[1] / w
            y_norm = landmark[2] / h
            features.extend([x_norm, y_norm])
        
        while len(features) < 42:
            features.extend([0.0, 0.0])
        
        return np.array(features[:42])
    
    def add_letter(self):
        """Add current prediction to text"""
        if self.current_prediction not in ["No hand", "Low confidence", "--"] and self.current_confidence > 0.8:
            self.recognized_text += self.current_prediction
            self.update_text_display()
    
    def add_space(self):
        """Add space to text"""
        self.recognized_text += " "
        self.update_text_display()
    
    def delete_last(self):
        """Delete last character"""
        if self.recognized_text:
            self.recognized_text = self.recognized_text[:-1]
            self.update_text_display()
    
    def update_text_display(self):
        """Update text display"""
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(1.0, self.recognized_text)
    
    def translate_text(self):
        """Translate text to selected language"""
        if not self.recognized_text:
            return
            
        target_lang = self.languages[self.lang_var.get()]
        
        if target_lang == 'en':
            self.translated_text = self.recognized_text
        else:
            try:
                result = self.translator.translate(self.recognized_text, dest=target_lang)
                self.translated_text = result.text
            except Exception as e:
                self.translated_text = f"Translation error: {e}"
        
        self.trans_display.delete(1.0, tk.END)
        self.trans_display.insert(1.0, self.translated_text)
    
    def clear_all(self):
        """Clear all text"""
        self.recognized_text = ""
        self.translated_text = ""
        self.text_display.delete(1.0, tk.END)
        self.trans_display.delete(1.0, tk.END)
    
    def quit_app(self):
        """Quit application"""
        self.stop_camera()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = ASLTranslatorApp(root)
    root.mainloop()

if __name__ == "__main__":
>>>>>>> ef90c81ce9fa93d6e1cf089de58be96be22e2dfd
    main()