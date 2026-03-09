import cv2
import mediapipe as mp
import math

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Adjusted for better detection
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,  # Slightly higher for accuracy
    min_tracking_confidence=0.6
)

# Finger landmarks (0-based indices for hand model)
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
THUMB_IP = 3  # Thumb interphalangeal joint
INDEX_DIP = 6  # Distal interphalangeal for index
MIDDLE_DIP = 10
RING_DIP = 14
PINKY_DIP = 18
WRIST = 0

def get_finger_states(landmarks, handedness_label):
    """
    Determine if each finger is extended or curled.
    Uses y-coordinates with buffer for robustness.
    handedness_label: 'Left' or 'Right' to adjust for mirroring.
    """
    h, w, _ = 640, 480, 3  # Assuming frame size; adjust if needed
    buffer = 0.03  # Looser threshold for detection (was tighter before)
    
    # Get landmark positions (normalized to pixel coords)
    thumb_tip = (int(landmarks.landmark[THUMB_TIP].x * w), int(landmarks.landmark[THUMB_TIP].y * h))
    thumb_ip = (int(landmarks.landmark[THUMB_IP].x * w), int(landmarks.landmark[THUMB_IP].y * h))
    index_tip = (int(landmarks.landmark[INDEX_TIP].x * w), int(landmarks.landmark[INDEX_TIP].y * h))
    index_dip = (int(landmarks.landmark[INDEX_DIP].x * w), int(landmarks.landmark[INDEX_DIP].y * h))
    middle_tip = (int(landmarks.landmark[MIDDLE_TIP].x * w), int(landmarks.landmark[MIDDLE_TIP].y * h))
    middle_dip = (int(landmarks.landmark[MIDDLE_DIP].x * w), int(landmarks.landmark[MIDDLE_DIP].y * h))
    ring_tip = (int(landmarks.landmark[RING_TIP].x * w), int(landmarks.landmark[RING_TIP].y * h))
    ring_dip = (int(landmarks.landmark[RING_DIP].x * w), int(landmarks.landmark[RING_DIP].y * h))
    pinky_tip = (int(landmarks.landmark[PINKY_TIP].x * w), int(landmarks.landmark[PINKY_TIP].y * h))
    pinky_dip = (int(landmarks.landmark[PINKY_DIP].x * w), int(landmarks.landmark[PINKY_DIP].y * h))
    
    # Thumb check (special: uses x-distance from wrist for extension)
    thumb_extended = False
    if handedness_label == 'Right':
        thumb_extended = thumb_tip[0] > thumb_ip[0] + (buffer * w)  # Thumb out to right
    else:  # Left
        thumb_extended = thumb_tip[0] < thumb_ip[0] - (buffer * w)  # Thumb out to left
    
    # Other fingers: extended if tip is above (lower y) DIP joint
    index_extended = index_tip[1] < index_dip[1] - (buffer * h)
    middle_extended = middle_tip[1] < middle_dip[1] - (buffer * h)
    ring_extended = ring_tip[1] < ring_dip[1] - (buffer * h)
    pinky_extended = pinky_tip[1] < pinky_dip[1] - (buffer * h)
    
    # Optional debug: Print states (uncomment if needed)
    # print(f"Thumb: {'Up' if thumb_extended else 'Down'}, Index: {'Up' if index_extended else 'Down'}, Middle: {'Up' if middle_extended else 'Down'}, Ring: {'Up' if ring_extended else 'Down'}, Pinky: {'Up' if pinky_extended else 'Down'}")
    
    return thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended

def get_gesture(thumb, index, middle, ring, pinky):
    """
    Classify gesture based on finger states.
    More robust logic with priorities.
    """
    extended_fingers = sum([thumb, index, middle, ring, pinky])
    
    if extended_fingers == 0:
        return "Fist ✊"
    elif extended_fingers >= 4:
        return "Open Palm ✋"
    elif thumb and index and not middle and not ring and not pinky:
        return "Thumbs Up 👍"
    elif index and middle and not thumb and not ring and not pinky:
        return "Victory ✌️"
    else:
        return "Unknown ❓"

# Capture from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process hands
    results = hands.process(rgb_frame)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Draw landmarks and connections
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get handedness label
            handedness_label = handedness.classification[0].label  # 'Left' or 'Right'
            
            # Get finger states
            thumb, index, middle, ring, pinky = get_finger_states(hand_landmarks, handedness_label)
            
            # Classify gesture
            gesture = get_gesture(thumb, index, middle, ring, pinky)
            
            # Draw label above hand
            h, w, _ = frame.shape
            label_y = int(hand_landmarks.landmark[WRIST].y * h) - 20
            cv2.putText(frame, f"{gesture} ({handedness_label})", (10, label_y if label_y > 30 else 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)
    
    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()