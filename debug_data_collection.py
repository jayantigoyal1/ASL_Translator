import cv2
import numpy as np
import os
from hand_detection import HandDetector

def debug_data_collection():
    """Debug version to find the issue"""
    
    print("=== DEBUG: Starting data collection debug ===")
    
    # Step 1: Test camera again
    print("Step 1: Testing camera...")
    camera_index = int(input("Which camera index worked in the test? (0, 1, or 2): ") or 0)
    
    print(f"Trying to open camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"ERROR: Camera {camera_index} failed to open!")
        print("Try running camera_test.py again to confirm which camera works")
        return
    
    print("✅ Camera opened successfully!")
    
    # Step 2: Test reading frames
    print("Step 2: Testing frame reading...")
    ret, img = cap.read()
    if not ret:
        print("ERROR: Can't read frames from camera!")
        cap.release()
        return
    
    print("✅ Frame reading works!")
    print(f"Frame size: {img.shape}")
    
    # Step 3: Test hand detector
    print("Step 3: Testing hand detector...")
    try:
        detector = HandDetector()
        print("✅ Hand detector created successfully!")
    except Exception as e:
        print(f"ERROR: Hand detector failed: {e}")
        cap.release()
        return
    
    # Step 4: Test the main loop
    print("Step 4: Testing main detection loop...")
    print("Show your hand to the camera. Press 'q' to quit this test.")
    
    frame_count = 0
    while frame_count < 100:  # Test for 100 frames max
        ret, img = cap.read()
        if not ret:
            print("ERROR: Lost camera connection!")
            break
        
        # Flip image
        img = cv2.flip(img, 1)
        
        # Try hand detection
        try:
            img = detector.find_hands(img, draw=True)
            lm_list = detector.find_position(img, draw=False)
            
            # Display info
            cv2.putText(img, f'Frame: {frame_count}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if len(lm_list) > 0:
                cv2.putText(img, f'Landmarks: {len(lm_list)}', (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, 'No hand detected', (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Debug - Data Collection Test', img)
            
        except Exception as e:
            print(f"ERROR in detection loop: {e}")
            break
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("User pressed 'q' - exiting test")
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    if frame_count > 0:
        print("✅ Main loop worked! The issue might be elsewhere.")
        print("Let's try the actual data collection now...")
        
        # Try a simple collection
        test_simple_collection(camera_index)
    else:
        print("❌ Main loop failed!")

def test_simple_collection(camera_index):
    """Test simple data collection"""
    print("\n=== Testing simple data collection ===")
    
    cap = cv2.VideoCapture(camera_index)
    detector = HandDetector()
    
    print("Simple collection test - press 's' to collect 5 samples, 'q' to quit")
    
    samples_collected = 0
    collecting = False
    
    while samples_collected < 5:
        ret, img = cap.read()
        if not ret:
            break
        
        img = cv2.flip(img, 1)
        img = detector.find_hands(img, draw=True)
        lm_list = detector.find_position(img, draw=False)
        
        # Display info
        cv2.putText(img, f'Samples: {samples_collected}/5', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if collecting:
            cv2.putText(img, 'COLLECTING!', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(img, "Press 's' to collect", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Collect if hand detected and collecting mode
        if len(lm_list) > 0 and collecting:
            print(f"Collected sample {samples_collected + 1}")
            samples_collected += 1
            collecting = False  # Stop collecting after each sample
        
        cv2.imshow('Simple Collection Test', img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            collecting = True
            print("Started collecting...")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"✅ Collected {samples_collected} samples successfully!")

if __name__ == "__main__":
    debug_data_collection()