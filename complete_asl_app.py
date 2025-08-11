import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.font as tkfont
import cv2
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image, ImageTk
from googletrans import Translator
from collections import deque
import os
from hand_detection import HandDetector

class ASLTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language to Text Translator")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2c3e50')

        # Components
        self.detector = HandDetector()
        self.translator = Translator()
        self.model = None
        self.label_encoder = None
        self.scaler = None

        # Camera state
        self.cap = None
        self.camera_running = False
        self.current_prediction = "No hand"
        self.current_confidence = 0.0

        # Prediction stability
        self.prediction_buffer = deque(maxlen=5)
        self.confidence_threshold = 0.8
        self.stable_prediction = None
        self.stable_count = 0
        self.stability_threshold = 3

        # Text state
        self.recognized_text = ""
        self.translated_text = ""

        # Languages (Punjabi added back)
        self.languages = {
            'English': 'en',
            'Hindi': 'hi',
            'Tamil': 'ta',
            'Telugu': 'te',
            'Bengali': 'bn',
            'Marathi': 'mr',
            'Gujarati': 'gu',
            'Kannada': 'kn',
            'Punjabi': 'pa'
        }

        self.load_model()
        self.create_gui()

    # ===== FONT HANDLING =====
    def _preferred_fonts_for_language(self, language):
        prefs = {
            'English': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'Hindi': ['Noto Sans Devanagari', 'Lohit Devanagari', 'Nirmala UI', 'Mangal'],
            'Tamil': ['Noto Sans Tamil', 'Lohit Tamil', 'Nirmala UI'],
            'Telugu': ['Noto Sans Telugu', 'Lohit Telugu', 'Nirmala UI'],
            'Bengali': ['Noto Sans Bengali', 'Lohit Bengali', 'Nirmala UI'],
            'Marathi': ['Noto Sans Devanagari', 'Lohit Devanagari', 'Nirmala UI'],
            'Gujarati': ['Noto Sans Gujarati', 'Lohit Gujarati', 'Nirmala UI'],
            'Kannada': ['Noto Sans Kannada', 'Lohit Kannada', 'Nirmala UI'],
            'Punjabi': ['Noto Sans Gurmukhi', 'Lohit Gurmukhi', 'Nirmala UI']
        }
        return prefs.get(language, prefs['English'])

    def _choose_installed_font(self, language, size=12):
        available = set(tkfont.families())
        for fam in self._preferred_fonts_for_language(language):
            if fam in available:
                return (fam, size)
        return ('Arial', size)

    def _on_language_changed(self, event=None):
        lang = self.lang_var.get()
        chosen_font = self._choose_installed_font(lang, size=12)
        self.trans_display.configure(font=chosen_font)
        if self.recognized_text.strip():
            self.translate_text()

    # ===== MODEL LOADING =====
    def load_model(self):
        model_files = [
            ("improved_asl_model.h5", "improved_label_encoder.pkl", "feature_scaler.pkl"),
            ("best_model.h5", "label_encoder.pkl", None),
            ("asl_model.h5", "label_encoder.pkl", None)
        ]
        for model_path, encoder_path, scaler_path in model_files:
            if os.path.exists(model_path) and os.path.exists(encoder_path):
                try:
                    print(f"📥 Loading model: {model_path}")
                    self.model = tf.keras.models.load_model(model_path)
                    print("✅ Model loaded successfully!")
                    with open(encoder_path, 'rb') as f:
                        self.label_encoder = pickle.load(f)
                    print("✅ Label encoder loaded!")
                    if scaler_path and os.path.exists(scaler_path):
                        with open(scaler_path, 'rb') as f:
                            self.scaler = pickle.load(f)
                        print("✅ Feature scaler loaded!")
                    else:
                        self.scaler = None
                    return
                except Exception as e:
                    print(f"❌ Error loading {model_path}: {e}")
        messagebox.showerror("Error", "No model found! Please ensure model files exist.")

    # ===== GUI CREATION =====
    def create_gui(self):
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill=tk.X)
        tk.Label(title_frame, text="Sign Language to Text",
                 font=('Arial', 28, 'bold'), bg='#2c3e50', fg='white').pack(expand=True)

        main_frame = tk.Frame(self.root, bg='#34495e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Camera side
        left_frame = tk.Frame(main_frame, bg='#34495e')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.camera_label = tk.Label(left_frame, bg='black', text="Camera Off\n\nClick Start Camera",
                                     fg='white', font=('Arial', 18))
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=(0, 20))
        cam_controls = tk.Frame(left_frame, bg='#34495e')
        cam_controls.pack(fill=tk.X, pady=(10, 0))
        tk.Button(cam_controls, text="Start Camera", command=self.start_camera,
                  bg='#27ae60', fg='white', font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=5)
        tk.Button(cam_controls, text="Stop Camera", command=self.stop_camera,
                  bg='#e74c3c', fg='white', font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=5)

        # Controls side
        right_frame = tk.Frame(main_frame, bg='#34495e', width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Prediction
        pred_frame = tk.LabelFrame(right_frame, text="Current Prediction",
                                   bg='#34495e', fg='white', font=('Arial', 14, 'bold'))
        pred_frame.pack(fill=tk.X, pady=(0, 20))
        self.pred_label = tk.Label(pred_frame, text="--", font=('Arial', 36, 'bold'),
                                   bg='#34495e', fg='#3498db')
        self.pred_label.pack(pady=10)
        self.conf_label = tk.Label(pred_frame, text="Confidence: 0%",
                                   bg='#34495e', fg='#bdc3c7')
        self.conf_label.pack()
        self.stability_label = tk.Label(pred_frame, text="Stability: 0/3",
                                        bg='#34495e', fg='#95a5a6')
        self.stability_label.pack()

        # Buttons
        btn_frame = tk.Frame(right_frame, bg='#34495e')
        btn_frame.pack(fill=tk.X, pady=10)
        tk.Button(btn_frame, text="Add Letter", command=self.add_letter,
                  bg='#3498db', fg='white', font=('Arial', 14, 'bold')).pack(fill=tk.X, pady=2)
        tk.Button(btn_frame, text="Add Space", command=self.add_space,
                  bg='#f39c12', fg='white', font=('Arial', 14, 'bold')).pack(fill=tk.X, pady=2)
        tk.Button(btn_frame, text="Delete Last", command=self.delete_last,
                  bg='#95a5a6', fg='white', font=('Arial', 14, 'bold')).pack(fill=tk.X, pady=2)

        # Confidence threshold
        th_frame = tk.LabelFrame(right_frame, text="Confidence Threshold",
                                 bg='#34495e', fg='white')
        th_frame.pack(fill=tk.X, pady=10)
        self.threshold_var = tk.DoubleVar(value=self.confidence_threshold)
        tk.Scale(th_frame, from_=0.5, to=0.95, resolution=0.05,
                 orient=tk.HORIZONTAL, variable=self.threshold_var,
                 bg='#34495e', fg='white', command=self.update_threshold).pack(fill=tk.X)
        self.threshold_label = tk.Label(th_frame, text=f"Current: {self.confidence_threshold:.2f}",
                                        bg='#34495e', fg='#bdc3c7')
        self.threshold_label.pack()

        # Language
        lang_frame = tk.LabelFrame(right_frame, text="Target Language",
                                   bg='#34495e', fg='white', font=('Arial', 14, 'bold'))
        lang_frame.pack(fill=tk.X, pady=10)
        self.lang_var = tk.StringVar(value="English")
        lang_combo = ttk.Combobox(lang_frame, textvariable=self.lang_var,
                                  values=list(self.languages.keys()), state="readonly")
        lang_combo.pack(fill=tk.X, padx=10, pady=5)
        lang_combo.bind('<<ComboboxSelected>>', self._on_language_changed)

        # Recognized text
        self.text_display = tk.Text(right_frame, height=4, font=('Arial', 12),
                                    bg='#ecf0f1', fg='#2c3e50', wrap=tk.WORD)
        self.text_display.pack(fill=tk.X, pady=5)

        # Translated text
        self.trans_display = tk.Text(right_frame, height=4,
                                     font=self._choose_installed_font("English", 12),
                                     bg='#d5dbdb', fg='#2c3e50', wrap=tk.WORD)
        self.trans_display.pack(fill=tk.BOTH, expand=True, pady=5)

        # Bottom buttons
        bottom = tk.Frame(self.root, bg='#2c3e50', height=60)
        bottom.pack(fill=tk.X)
        tk.Button(bottom, text="Translate Text", command=self.translate_text,
                  bg='#9b59b6', fg='white', font=('Arial', 16, 'bold')).pack(side=tk.LEFT, padx=10)
        tk.Button(bottom, text="Clear All", command=self.clear_all,
                  bg='#e67e22', fg='white', font=('Arial', 16, 'bold')).pack(side=tk.LEFT, padx=10)
        tk.Button(bottom, text="Quit", command=self.quit_app,
                  bg='#c0392b', fg='white', font=('Arial', 16, 'bold')).pack(side=tk.RIGHT, padx=10)

    # ===== CAMERA LOGIC =====
    def update_threshold(self, value):
        self.confidence_threshold = float(value)
        self.threshold_label.configure(text=f"Current: {self.confidence_threshold:.2f}")

    def start_camera(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded!")
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera!")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.camera_running = True
        self.update_camera()

    def stop_camera(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
        self.camera_label.configure(image="", text="Camera Off\n\nClick Start Camera")
        self.stable_prediction = None
        self.stable_count = 0
        self.prediction_buffer.clear()

    def extract_features_improved(self, lm_list, img_shape):
        if len(lm_list) == 0:
            return None
        h, w, _ = img_shape
        features = []
        for landmark in lm_list:
            features.extend([landmark[1] / w, landmark[2] / h])
        while len(features) < 42:
            features.extend([0.0, 0.0])
        features = np.array(features[:42], dtype=np.float32)
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1))[0]
        return features

    def predict_gesture_improved(self, features):
        pred = self.model.predict(features.reshape(1, -1), verbose=0)[0]
        idx = np.argmax(pred)
        return self.label_encoder.classes_[idx], pred[idx]

    def stabilize_prediction(self, gesture, confidence):
        if confidence < self.confidence_threshold:
            return None, False
        if gesture == self.stable_prediction:
            self.stable_count += 1
        else:
            self.stable_prediction = gesture
            self.stable_count = 1
        return gesture if self.stable_count >= self.stability_threshold else None, self.stable_count >= self.stability_threshold

    def update_camera(self):
        if not self.camera_running:
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = self.detector.find_hands(frame, draw=True)
            landmarks = self.detector.find_position(frame, draw=False)
            current_prediction = "No hand detected"
            confidence = 0.0
            is_stable = False
            if landmarks:
                features = self.extract_features_improved(landmarks, frame.shape)
                if features is not None:
                    gesture, conf = self.predict_gesture_improved(features)
                    stable_gesture, is_stable = self.stabilize_prediction(gesture, conf)
                    current_prediction = stable_gesture if stable_gesture else f"{gesture} (unstable)"
                    confidence = conf
            else:
                self.stable_prediction = None
                self.stable_count = 0
            self.current_prediction = current_prediction
            self.current_confidence = confidence
            self.pred_label.configure(text=current_prediction)
            self.conf_label.configure(text=f"Confidence: {confidence:.1%}")
            self.stability_label.configure(text=f"Stability: {self.stable_count}/{self.stability_threshold}",
                                           fg='#27ae60' if is_stable else '#95a5a6')
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(cv2.resize(rgb, (500, 375))))
            self.camera_label.configure(image=photo, text="")
            self.camera_label.image = photo
        self.root.after(50, self.update_camera)

    # ===== TEXT & TRANSLATION =====
    def add_letter(self):
        if (self.current_prediction not in ["No hand detected", "Feature extraction failed", "--"]
            and self.stable_count >= self.stability_threshold
            and self.current_confidence > self.confidence_threshold):
            letter = self.current_prediction.split(" (")[0]
            self.recognized_text += letter
            self.update_text_display()
            self.stable_prediction = None
            self.stable_count = 0
            self.prediction_buffer.clear()

    def add_space(self):
        self.recognized_text += " "
        self.update_text_display()

    def delete_last(self):
        if self.recognized_text:
            self.recognized_text = self.recognized_text[:-1]
            self.update_text_display()

    def update_text_display(self):
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(1.0, self.recognized_text)

    def translate_text(self):
        if not self.recognized_text.strip():
            messagebox.showwarning("Warning", "No text to translate!")
            return
        lang_code = self.languages[self.lang_var.get()]
        self.trans_display.configure(font=self._choose_installed_font(self.lang_var.get(), 12))
        if lang_code == 'en':
            self.translated_text = self.recognized_text
        else:
            try:
                self.translated_text = self.translator.translate(self.recognized_text, dest=lang_code).text
            except Exception as e:
                self.translated_text = f"Translation error: {e}"
        self.trans_display.delete(1.0, tk.END)
        self.trans_display.insert(1.0, self.translated_text)

    def clear_all(self):
        self.recognized_text = ""
        self.translated_text = ""
        self.text_display.delete(1.0, tk.END)
        self.trans_display.delete(1.0, tk.END)
        self.stable_prediction = None
        self.stable_count = 0
        self.prediction_buffer.clear()

    def quit_app(self):
        self.stop_camera()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = ASLTranslatorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
