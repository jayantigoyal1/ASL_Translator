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
    main()