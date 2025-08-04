<<<<<<< HEAD
import cv2

def test_camera():
    """Simple camera test to check if camera works"""
    print("Testing camera access...")
    
    # Try different camera indices
    for i in range(3):
        print(f"Trying camera index {i}...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            print(f"Camera {i} opened successfully!")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"Camera {i} is working! Press 'q' to quit.")
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    cv2.imshow(f'Camera Test - Index {i}', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                cap.release()
                cv2.destroyAllWindows()
                return i  # Return working camera index
            else:
                print(f"Camera {i} opened but can't read frames")
        else:
            print(f"Camera {i} failed to open")
        
        cap.release()
    
    print("No working camera found!")
    return None

if __name__ == "__main__":
    working_camera = test_camera()
    if working_camera is not None:
        print(f"\nYour working camera index is: {working_camera}")
        print("Use this index in your main code!")
    else:
        print("\nTroubleshooting steps:")
        print("1. Close any apps using camera (Zoom, Teams, etc.)")
        print("2. Check camera permissions in Windows settings")
=======
import cv2

def test_camera():
    """Simple camera test to check if camera works"""
    print("Testing camera access...")
    
    # Try different camera indices
    for i in range(3):
        print(f"Trying camera index {i}...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            print(f"Camera {i} opened successfully!")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"Camera {i} is working! Press 'q' to quit.")
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    cv2.imshow(f'Camera Test - Index {i}', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                cap.release()
                cv2.destroyAllWindows()
                return i  # Return working camera index
            else:
                print(f"Camera {i} opened but can't read frames")
        else:
            print(f"Camera {i} failed to open")
        
        cap.release()
    
    print("No working camera found!")
    return None

if __name__ == "__main__":
    working_camera = test_camera()
    if working_camera is not None:
        print(f"\nYour working camera index is: {working_camera}")
        print("Use this index in your main code!")
    else:
        print("\nTroubleshooting steps:")
        print("1. Close any apps using camera (Zoom, Teams, etc.)")
        print("2. Check camera permissions in Windows settings")
>>>>>>> ef90c81ce9fa93d6e1cf089de58be96be22e2dfd
        print("3. Try reconnecting external camera if using one")