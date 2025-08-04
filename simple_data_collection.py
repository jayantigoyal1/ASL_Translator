import cv2
import numpy as np
import os
import time
from hand_detection import HandDetector

class SimpleDataCollector:
    def __init__(self, data_dir="asl_data"):
        self.data_dir = data_dir
        self.detector = HandDetector()
        
        # All ASL letters including J and Z
        self.gestures = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        
        # Create directories
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        for gesture in self.gestures:
            gesture_dir = os.path.join(self.data_dir, gesture)
            if not os.path.exists(gesture_dir):
                os.makedirs(gesture_dir)
    
    def collect_single_gesture(self, gesture, samples=100, camera_index=0):
        """Collect data for one gesture"""
        
        print(f"\n=== Collecting data for gesture: {gesture} ===")
        
        # Special instructions for motion letters
        if gesture == 'J':
            print("📍 For J: Start with pinky up (like 'I'), then hook it toward you.")
            print("   Hold the FINAL hooked position while collecting.")
        elif gesture == 'Z':
            print("📍 For Z: Point your index finger straight out.")
            print("   We'll capture the pointing position, not the zigzag motion.")
        
        input("Press Enter when you're ready to start...")
        
        # Open camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"ERROR: Could not open camera {camera_index}")
            return False
        
        print("Camera opened! Instructions:")
        print("- Position your hand to make the gesture")
        print("- Press 's' to START collecting")
        print("- Press 'q' to QUIT")
        
        collected = 0
        collecting = False
        
        while collected < samples:
            ret, img = cap.read()
            if not ret:
                print("ERROR: Could not read from camera")
                break
            
            # Flip for natural interaction
            img = cv2.flip(img, 1)
            
            # Find hands
            img = self.detector.find_hands(img, draw=True)
            lm_list = self.detector.find_position(img, draw=False)
            
            # Display information
            cv2.putText(img, f'Gesture: {gesture}', (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(img, f'Collected: {collected}/{samples}', (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if collecting:
                cv2.putText(img, 'COLLECTING...', (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.putText(img, "Press 's' to start", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Show hand detection status
            if len(lm_list) > 0:
                cv2.putText(img, f'Hand: {len(lm_list)} landmarks', (10, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Collect data if we're in collecting mode
                if collecting:
                    features = self.extract_features(lm_list, img.shape)
                    filename = os.path.join(self.data_dir, gesture, f'{gesture}_{collected:03d}.npy')
                    np.save(filename, features)
                    collected += 1
                    
                    # Brief pause between collections
                    time.sleep(0.15)
                    
                    if collected % 10 == 0:  # Progress update every 10 samples
                        print(f"Progress: {collected}/{samples} samples collected")
            else:
                cv2.putText(img, 'No hand detected', (10, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow(f'Collecting ASL {gesture}', img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if len(lm_list) > 0:
                    collecting = True
                    print("Started collecting! Keep your hand steady...")
                else:
                    print("Please show your hand first!")
            elif key == ord('q'):
                print("Collection stopped by user")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"✅ Successfully collected {collected} samples for gesture '{gesture}'")
        print(f"Data saved in: {os.path.join(self.data_dir, gesture)}")
        
        return collected == samples
    
    def extract_features(self, lm_list, img_shape):
        """Extract normalized features from landmarks"""
        features = []
        h, w, _ = img_shape
        
        # Normalize coordinates relative to image size
        for landmark in lm_list:
            x_norm = landmark[1] / w
            y_norm = landmark[2] / h
            features.extend([x_norm, y_norm])
        
        # Ensure exactly 42 features (21 landmarks * 2 coordinates)
        while len(features) < 42:
            features.extend([0.0, 0.0])
        
        return np.array(features[:42])

def main():
    """Main function"""
    print("🤖 ASL Data Collection System")
    print("=" * 40)
    
    collector = SimpleDataCollector()
    
    # Get camera index
    camera_index = int(input("Which camera index worked in the test? (0, 1, or 2): ") or 0)
    
    # Show available gestures
    print(f"\nAvailable gestures: {collector.gestures}")
    
    while True:
        print("\nOptions:")
        print("1. Collect single gesture")
        print("2. Collect multiple gestures")
        print("3. Exit")
        
        choice = input("Choose option (1-3): ").strip()
        
        if choice == '1':
            gesture = input("Enter gesture letter (A-Z): ").upper().strip()
            if gesture in collector.gestures:
                samples = int(input("Number of samples (default 100): ") or 100)
                collector.collect_single_gesture(gesture, samples, camera_index)
            else:
                print(f"Invalid gesture! Choose from: {collector.gestures}")
        
        elif choice == '2':
            start_letter = input("Start from which letter? (A-Z): ").upper().strip()
            samples = int(input("Samples per gesture (default 100): ") or 100)
            
            if start_letter in collector.gestures:
                start_idx = collector.gestures.index(start_letter)
                for gesture in collector.gestures[start_idx:]:
                    print(f"\n{'='*50}")
                    success = collector.collect_single_gesture(gesture, samples, camera_index)
                    if not success:
                        break
                    
                    continue_choice = input("\nContinue to next gesture? (y/n): ").lower()
                    if continue_choice != 'y':
                        break
            else:
                print(f"Invalid starting letter! Choose from: {collector.gestures}")
        
        elif choice == '3':
            print("Goodbye! 👋")
            break
        
        else:
            print("Invalid choice! Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()