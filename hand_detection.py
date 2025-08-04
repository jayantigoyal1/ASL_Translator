import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        """
        Initialize the HandDetector
        
        Args:
            mode: If set to false, the solution treats the input images as a video stream
            max_hands: Maximum number of hands to detect
            detection_confidence: Minimum detection confidence
            tracking_confidence: Minimum tracking confidence
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def find_hands(self, img, draw=True):
        """
        Find hands in the image
        
        Args:
            img: Input image
            draw: Whether to draw hand landmarks
            
        Returns:
            img: Image with or without hand landmarks drawn
        """
        # Convert BGR to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        # Draw hand landmarks
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
        return img
    
    def find_position(self, img, hand_no=0, draw=True):
        """
        Find hand landmark positions
        
        Args:
            img: Input image
            hand_no: Hand number (0 for first hand, 1 for second)
            draw: Whether to draw landmarks
            
        Returns:
            lm_list: List of landmark positions [id, x, y]
        """
        lm_list = []
        
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                my_hand = self.results.multi_hand_landmarks[hand_no]
                
                for id, lm in enumerate(my_hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
                    
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        return lm_list

def main():
    """
    Test the hand detection system
    """
    # Initialize camera
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        # Find hands
        img = detector.find_hands(img)
        lm_list = detector.find_position(img)
        
        # Display landmark count
        if len(lm_list) != 0:
            cv2.putText(img, f'Landmarks: {len(lm_list)}', (10, 50), 
                       cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
        # Display the image
        cv2.imshow('Hand Detection', img)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()