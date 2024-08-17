import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize pyautogui
screen_width, screen_height = pyautogui.size()

# Smoothening factor
smoothening = 9
plocx, plocy = 0, 0
clocx, clocy = 0, 0

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    _, frame = cap.read()
    
    # Flip frame horizontally
    frame = cv2.flip(frame, 1)
    
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame
    results = hands.process(rgb_frame)
    
    # Draw hand landmarks on frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get hand landmarks
            landmarks = hand_landmarks.landmark
            
            # Get index finger tip
            index_tip = landmarks[8]
            index_x = int(index_tip.x * frame.shape[1])
            index_y = int(index_tip.y * frame.shape[0])
            
            # Draw circle at index finger tip
            cv2.circle(frame, (index_x, index_y), 15, (0, 255, 255), -1)
            
            # Convert index finger tip to screen coordinates
            screen_x = (screen_width / frame.shape[1]) * index_x
            screen_y = (screen_height / frame.shape[0]) * index_y
            
            # Smoothen mouse movement
            clocx = plocx + (screen_x - plocx) / smoothening
            clocy = plocy + (screen_y - plocy) / smoothening
            
            # Move mouse
            pyautogui.moveTo(clocx, clocy)
            
            # Update previous location
            plocx, plocy = clocx, clocy
            
            # Get thumb tip
            thumb_tip = landmarks[4]
            thumb_x = int(thumb_tip.x * frame.shape[1])
            thumb_y = int(thumb_tip.y * frame.shape[0])
            
            # Draw circle at thumb tip
            cv2.circle(frame, (thumb_x, thumb_y), 15, (0, 255, 255), -1)
            
            # Check if thumb and index finger are close
            if abs(index_y - thumb_y) < 70:
                # Click mouse
                pyautogui.click()
                pyautogui.sleep(1)
            
            # Get pinky finger tip
            pinky_tip = landmarks[20]
            pinky_x = int(pinky_tip.x * frame.shape[1])
            pinky_y = int(pinky_tip.y * frame.shape[0])
            
            # Draw circle at pinky finger tip
            cv2.circle(frame, (pinky_x, pinky_y), 15, (0, 255, 255), -1)
            
            # Check if pinky finger is up
            if pinky_y < index_y:
                # Increase volume
                pyautogui.press('volumeup')
                pyautogui.sleep(0.5)
            
            # Get ring finger tip
            ring_tip = landmarks[16]
            ring_x = int(ring_tip.x * frame.shape[1])
            ring_y = int(ring_tip.y * frame.shape[0])
            
            # Draw circle at ring finger tip
            cv2.circle(frame, (ring_x, ring_y), 15, (0, 255, 255), -1)
            
            # Check if ring finger is up
            if ring_y < index_y:
                # Decrease volume
                pyautogui.press('volumedown')
                pyautogui.sleep(0.5)
    
    # Display frame
    cv2.imshow('Virtual Mouse', frame)
    
    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
