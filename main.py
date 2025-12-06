# ==================== IMPORTS ====================
import cv2                      # OpenCV for video capture and image processing
import pyautogui                # For simulating keyboard/mouse inputs to control the game
from time import time           # For calculating FPS (frames per second)
from math import hypot          # For calculating Euclidean distance between points
import mediapipe as mp          # Google's MediaPipe for pose detection
import matplotlib.pyplot as plt # For displaying images (used in display mode)


# ==================== MEDIAPIPE SETUP ====================
# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose

# Pose detector for static images (higher accuracy, slower)
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)

# Pose detector for video (optimized for real-time, uses tracking between frames)
pose_video = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)

# Drawing utilities for visualizing pose landmarks
mp_drawing = mp.solutions.drawing_utils


# Detects human body pose landmarks in an image using MediaPipe Pose.
# Converts image to RGB, processes it, and optionally draws landmarks on the output.
# Returns the output image and pose detection results.
def detectPose(image, pose, draw=False, display=False):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    if results.pose_landmarks and draw:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255),
                                                                               thickness=3, circle_radius=3),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(49,125,237),
                                                                               thickness=2, circle_radius=2))

    if display:
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        plt.show()
        return output_image, results
    else:
        return output_image, results


# Checks if the user's hands are joined by calculating the Euclidean distance
# between left and right wrist landmarks. Returns 'Hands Joined' if distance < 130px.
def checkHandsJoined(image, results, draw=False, display=False):
    height, width, _ = image.shape
    output_image = image.copy()
    
    # Check if pose landmarks exist
    if not results.pose_landmarks:
        return output_image, 'Hands Not Joined'
    
    left_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height)

    right_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width,
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height)
    
    euclidean_distance = int(hypot(left_wrist_landmark[0] - right_wrist_landmark[0],
                                   left_wrist_landmark[1] - right_wrist_landmark[1]))

    if euclidean_distance < 130:
        hand_status = 'Hands Joined'
        color = (0, 255, 0)
    else:
        hand_status = 'Hands Not Joined'
        color = (0, 0, 255)
        
    if draw:
        cv2.putText(output_image, hand_status, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        cv2.putText(output_image, f'Distance: {euclidean_distance}', (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        
    if display:
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        plt.show()
        return output_image, hand_status
    else:
        return output_image, hand_status
    

# Determines the horizontal position of the user (Left, Right, or Center)
# by analyzing shoulder landmark positions relative to the frame center.
def checkLeftRight(image, results, draw=False, display=False):
    horizontal_position = None
    height, width, _ = image.shape
    output_image = image.copy()
    
    # Check if pose landmarks exist
    if not results.pose_landmarks:
        return output_image, 'Center'
    
    left_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width)
    right_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width)
    
    if (right_x <= width//2 and left_x <= width//2):
        horizontal_position = 'Left'
    elif (right_x >= width//2 and left_x >= width//2):
        horizontal_position = 'Right'
    elif (right_x >= width//2 and left_x <= width//2):
        horizontal_position = 'Center'
        
    if draw:
        cv2.putText(output_image, horizontal_position, (5, height - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        cv2.line(output_image, (width//2, 0), (width//2, height), (255, 255, 255), 2)
        
    if display:
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        plt.show()
        return output_image, horizontal_position
    else:
        return output_image, horizontal_position


# Detects if the user is jumping, crouching, or standing by comparing
# the average shoulder Y-position against a calibrated middle reference point (MID_Y).
def checkJumpCrouch(image, results, MID_Y=250, draw=False, display=False):
    height, width, _ = image.shape
    output_image = image.copy()
    
    # Check if pose landmarks exist
    if not results.pose_landmarks:
        return output_image, 'Standing'
    
    left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)
    right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)
    actual_mid_y = (right_y + left_y) // 2
    lower_bound = MID_Y-15
    upper_bound = MID_Y+100
    
    if (actual_mid_y < lower_bound):
        posture = 'Jumping'
    elif (actual_mid_y > upper_bound):
        posture = 'Crouching'
    else:
        posture = 'Standing'
        
    if draw:
        cv2.putText(output_image, posture, (5, height - 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        cv2.line(output_image, (0, MID_Y),(width, MID_Y),(255, 255, 255), 2)
        
    if display:
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        plt.show()
        return output_image, posture
    else:
        return output_image, posture
    

# ==================== VIDEO CAPTURE SETUP ====================
video = cv2.VideoCapture(0)  # Open default webcam (index 0)
video.set(3, 1280)           # Set video width to 1280 pixels
video.set(4, 960)            # Set video height to 960 pixels

# ==================== GAME STATE VARIABLES ====================
time1 = 0                    # Previous frame timestamp for FPS calculation
game_started = False         # Flag to track if game has started
x_pos_index = 1              # Horizontal lane position (0=Left, 1=Center, 2=Right)
y_pos_index = 1              # Vertical position state (1=Standing, 0=Crouching, 2=Jumping)
MID_Y = None                 # Calibrated Y-position for standing (set when game starts)
counter = 0                  # Counter for consecutive frames with hands joined
num_of_frames = 10           # Number of frames hands must be joined to trigger action

# ==================== MAIN GAME LOOP ====================
while video.isOpened():
    # Read a frame from the webcam
    ret, frame = video.read()
    if not ret:
        continue
    
    # Flip frame horizontally for mirror effect (more intuitive for user)
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    
    # Detect pose landmarks in the current frame
    frame, results = detectPose(frame, pose_video, draw=game_started)
    
    # Only process if pose landmarks are detected
    if results.pose_landmarks:
        if game_started:
            # ===== HORIZONTAL MOVEMENT CONTROL =====
            # Check user's horizontal position and simulate left/right key presses
            frame, horizontal_position = checkLeftRight(frame, results, draw=True)
            
            # Move left if user is on the left side or returning to center from right
            if (horizontal_position=='Left' and x_pos_index!=0) or (horizontal_position=='Center' and x_pos_index==2):
                pyautogui.press('left')
                x_pos_index -= 1               
            # Move right if user is on the right side or returning to center from left
            elif (horizontal_position=='Right' and x_pos_index!=2) or (horizontal_position=='Center' and x_pos_index==0):
                pyautogui.press('right')
                x_pos_index += 1
            
        else:
            # ===== PRE-GAME INSTRUCTIONS =====
            # Display instruction text with green background before game starts
            text = "JOIN BOTH HANDS TO START THE GAME."
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            font_thickness = 2
            text_color = (255, 255, 255)
            bg_color = (0, 255, 0)
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            x, y = 10, frame_height - 20
            cv2.rectangle(frame, (x - 10, y - text_height - 10), (x + text_width + 10, y + baseline + 10), bg_color, -1)
            cv2.putText(frame, text, (x, y), font, font_scale, text_color, font_thickness)

        
        # ===== HANDS JOINED DETECTION (START GAME / PAUSE) =====
        # Check if user's hands are joined together
        if checkHandsJoined(frame, results)[1] == 'Hands Joined':
            counter += 1  # Increment counter for consecutive frames
            
            # Trigger action after hands joined for required number of frames
            if counter == num_of_frames:
                if not(game_started):
                    # START GAME: Calibrate standing position and click to start
                    game_started = True
                    # Calculate the user's standing shoulder Y-position as reference
                    left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height)
                    right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height)
                    MID_Y = (right_y + left_y) // 2
                    # Click to start the game (adjust coordinates for your screen)
                    pyautogui.click(x=1300, y=800, button='left')
                else:
                    # DURING GAME: Press space (e.g., to use hoverboard/powerup)
                    pyautogui.press('space')
                counter = 0  # Reset counter after action
        else:
            counter = 0  # Reset counter if hands not joined
            
        # ===== VERTICAL MOVEMENT CONTROL (JUMP / CROUCH) =====
        # Only check jump/crouch after MID_Y is calibrated (game started)
        if MID_Y:
            frame, posture = checkJumpCrouch(frame, results, MID_Y, draw=True)
            
            # Jump: Press up arrow when user jumps from standing position
            if posture == 'Jumping' and y_pos_index == 1:
                pyautogui.press('up')
                y_pos_index += 1
            # Crouch: Press down arrow when user crouches from standing position
            elif posture == 'Crouching' and y_pos_index == 1:
                pyautogui.press('down')
                y_pos_index -= 1
            # Reset to standing state when user returns to standing position
            elif posture == 'Standing' and y_pos_index != 1:
                y_pos_index = 1
        
    else:
        # Reset counter if no pose detected
        counter = 0
        
    # ===== FPS CALCULATION AND DISPLAY =====
    time2 = time()
    if (time2 - time1) > 0:
        frames_per_second = 1.0 / (time2 - time1)
        cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    time1 = time2
    
    # Display the frame in a window
    cv2.imshow('Subway Surfers with Pose Detection', frame)
    
    # Check for ESC key (27) to exit the loop
    k = cv2.waitKey(1) & 0xFF    
    if(k == 27):
        break

# ==================== CLEANUP ====================
# Release webcam and close all OpenCV windows
video.release()
cv2.destroyAllWindows()