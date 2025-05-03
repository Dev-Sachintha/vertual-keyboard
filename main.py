import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np
import time

# --- Configuration ---
WEBCAM_INDEX = 0
SHOW_WEBCAM_FEED = True  # Set to False to hide webcam window (still runs)

# --- Screen Dimensions ---
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
print(f"Screen dimensions: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

# --- Frame Reduction & Smoothing (Mouse Movement - Right Hand) ---
FRAME_REDUCTION_X = 0.15  # % of screen width to ignore near edges
FRAME_REDUCTION_Y = 0.20  # % of screen height to ignore near edges
SMOOTHING = 7  # Higher value = smoother but more laggy movement

# --- Gesture Thresholds & Cooldowns ---
# -- Left Hand (Mode Control & Clicks *when not in KB mode*) --
FIST_DISTANCE_THRESHOLD = 60  # Avg distance of fingertips to wrist for fist (tune)
LEFT_CLICK_DISTANCE_THRESHOLD = 30  # Pixel distance for Left Click (Index <-> Thumb)
RIGHT_CLICK_DISTANCE_THRESHOLD = 30  # Pixel distance for Right Click (Middle <-> Thumb)
ACTION_COOLDOWN = 0.35  # Cooldown for mouse clicks OR mode toggle

# -- Visual Keyboard (Right Hand *when in KB mode*) --
VISUAL_KEY_PRESS_COOLDOWN = 0.4  # Min seconds between visual key presses (debounce)

# --- MediaPipe Initialization ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Detect up to two hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# --- State Variables ---
# Movement (Right Hand - Mouse Mode)
prev_x, prev_y = 0, 0
current_x, current_y = 0, 0
# Actions (Left Hand)
last_action_time = 0
# Keyboard Mode
keyboard_mode_active = False
last_key_press_time = 0
last_mode_toggle_time = 0
pressed_key_char = None  # Track the visually pressed key for highlighting

# --- Visual Keyboard Layout Definition ---
keys = [
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "<-"],  # Added Backspace
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "?"],
    ["SPACE"],  # Added Space
    # Enter might be better as a left-hand gesture or a larger key
]

key_width = 55
key_height = 55
key_spacing = 10
start_x = 20
start_y = 50

keyboard_layout = []
for i, row in enumerate(keys):
    row_width = len(row) * (key_width + key_spacing) - key_spacing
    current_start_x = (
        start_x
        + ((len(keys[0]) * (key_width + key_spacing) - key_spacing) - row_width) // 2
    )  # Center rows roughly
    if row[0] == "SPACE":  # Make space bar wider
        key_width_space = key_width * 5 + key_spacing * 4
        x = (
            start_x
            + (
                (len(keys[0]) * (key_width + key_spacing) - key_spacing)
                - key_width_space
            )
            // 2
        )
        y = start_y + i * (key_height + key_spacing)
        keyboard_layout.append(
            {"x": x, "y": y, "w": key_width_space, "h": key_height, "char": "SPACE"}
        )
    else:
        for j, key_char in enumerate(row):
            x = current_start_x + j * (key_width + key_spacing)
            y = start_y + i * (key_height + key_spacing)
            keyboard_layout.append(
                {"x": x, "y": y, "w": key_width, "h": key_height, "char": key_char}
            )

# --- Text Display Area (for visual keyboard output preview) ---
typed_text_preview = ""  # Preview on the webcam feed only
text_display_area_y = 15  # Y position relative to top for the text preview

# --- Webcam Initialization ---
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open webcam {WEBCAM_INDEX}")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Use higher res if available
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Webcam resolution: {frame_width}x{frame_height}")

print("Starting Two-Handed Virtual Control...")
print("- Right Hand Index: Move Cursor (Mouse Mode)")
print("- Left Hand Index+Thumb Pinch: Left Click (Mouse Mode)")
print("- Left Hand Middle+Thumb Pinch: Right Click (Mouse Mode)")
print("- Left Hand Fist: Toggle Keyboard Mode ON/OFF")
print("--- Keyboard Mode ON ---")
print("  - Right Hand Index Hover: Type on Visual Keyboard")
print("  - Visual Keys: '<-': Backspace, 'SPACE': Spacebar")
print("Press 'q' in the webcam window to quit.")


# --- Helper Functions ---
def calculate_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def get_landmark_pixel(landmark, frame_w, frame_h):
    if landmark.x < 0 or landmark.x > 1 or landmark.y < 0 or landmark.y > 1:
        return None  # Landmark out of bounds
    return int(landmark.x * frame_w), int(landmark.y * frame_h)


def draw_keyboard(img, layout, highlight_key_char=None, cooldown_active=False):
    overlay = img.copy()
    alpha = 0.6  # Transparency factor for keys

    for key in layout:
        x, y, w, h = key["x"], key["y"], key["w"], key["h"]
        char = key["char"]
        display_char = char if char != "<-" else "<"  # Display abbreviation
        if char == "SPACE":
            display_char = "____"

        # Key background with transparency
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (220, 220, 220), cv2.FILLED)

        # Highlight if pressed (and not on cooldown)
        if highlight_key_char and highlight_key_char == char and not cooldown_active:
            cv2.rectangle(
                overlay, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED
            )  # Green highlight

        # Key border
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (50, 50, 50), 2)

        # Key character
        font_scale = 1.0 if len(display_char) <= 1 else 0.7
        text_size = cv2.getTextSize(
            display_char, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
        )[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(
            overlay,
            display_char,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            2,
        )

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)  # Apply overlay


# --- Main Loop ---
try:
    while True:
        current_time = time.time()
        success, frame = cap.read()
        if not success:
            print("Warning: Failed to grab frame.")
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False  # Performance optimization
        results = hands_detector.process(frame_rgb)
        frame_rgb.flags.writeable = True
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        right_hand_landmarks = None
        left_hand_landmarks = None
        right_hand_detected = False
        left_hand_detected = False

        # --- Separate Left and Right Hand Landmarks ---
        if results.multi_hand_landmarks and results.multi_handedness:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[i].classification[0]
                hand_label = handedness.label
                # Sometimes confidence is low, label might be wrong, check wrist pos?
                # Simple check: Left hand usually on left side of screen, Right on right
                wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                if hand_label == "Right":  # Initial detection
                    right_hand_landmarks = hand_landmarks
                    right_hand_detected = True
                elif hand_label == "Left":
                    left_hand_landmarks = hand_landmarks
                    left_hand_detected = True
                # Optional: Re-assign if label seems wrong based on wrist x-position
                # elif wrist_x > 0.5 and not right_hand_detected: # Likely Right but detected as Left
                #     right_hand_landmarks = hand_landmarks
                #     right_hand_detected = True
                # elif wrist_x <= 0.5 and not left_hand_detected: # Likely Left but detected as Right
                #     left_hand_landmarks = hand_landmarks
                #     left_hand_detected = True

                # --- Draw Landmarks (Optional) ---
                if SHOW_WEBCAM_FEED:
                    landmark_color = (
                        (121, 22, 76) if hand_label == "Right" else (76, 121, 22)
                    )  # Purple / Green
                    connection_color = (
                        (250, 44, 250) if hand_label == "Right" else (44, 250, 44)
                    )
                    mp_drawing.draw_landmarks(
                        frame_bgr,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(
                            color=landmark_color, thickness=1, circle_radius=3
                        ),
                        mp_drawing.DrawingSpec(
                            color=connection_color, thickness=2, circle_radius=2
                        ),
                    )

        # --- Draw Visual Keyboard If Mode Active ---
        visual_key_cooldown_active = (
            current_time < last_key_press_time + VISUAL_KEY_PRESS_COOLDOWN
        )
        if keyboard_mode_active and SHOW_WEBCAM_FEED:
            draw_keyboard(
                frame_bgr, keyboard_layout, pressed_key_char, visual_key_cooldown_active
            )

        # --- Process Left Hand (Mode Control & Clicks) ---
        is_fist = False
        if left_hand_landmarks:
            try:
                # Get landmarks
                wrist_left = left_hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                thumb_tip_left = left_hand_landmarks.landmark[
                    mp_hands.HandLandmark.THUMB_TIP
                ]
                index_tip_left = left_hand_landmarks.landmark[
                    mp_hands.HandLandmark.INDEX_FINGER_TIP
                ]
                middle_tip_left = left_hand_landmarks.landmark[
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                ]
                ring_tip_left = left_hand_landmarks.landmark[
                    mp_hands.HandLandmark.RING_FINGER_TIP
                ]
                pinky_tip_left = left_hand_landmarks.landmark[
                    mp_hands.HandLandmark.PINKY_TIP
                ]

                # Pixel coordinates
                wrist_px = get_landmark_pixel(wrist_left, frame_width, frame_height)
                thumb_px = get_landmark_pixel(thumb_tip_left, frame_width, frame_height)
                index_px = get_landmark_pixel(index_tip_left, frame_width, frame_height)
                middle_px = get_landmark_pixel(
                    middle_tip_left, frame_width, frame_height
                )
                ring_px = get_landmark_pixel(ring_tip_left, frame_width, frame_height)
                pinky_px = get_landmark_pixel(pinky_tip_left, frame_width, frame_height)

                # Ensure all landmarks were converted correctly
                if None not in [
                    wrist_px,
                    thumb_px,
                    index_px,
                    middle_px,
                    ring_px,
                    pinky_px,
                ]:
                    # --- Fist Detection ---
                    fingertip_landmarks = [index_px, middle_px, ring_px, pinky_px]
                    avg_fingertip_dist = np.mean(
                        [calculate_distance(p, wrist_px) for p in fingertip_landmarks]
                    )

                    fist_color = (0, 255, 0)  # Green - not fist
                    if avg_fingertip_dist < FIST_DISTANCE_THRESHOLD:
                        is_fist = True
                        fist_color = (0, 0, 255)  # Red - fist detected
                        # Toggle Keyboard Mode (with cooldown)
                        if current_time - last_mode_toggle_time > ACTION_COOLDOWN:
                            keyboard_mode_active = not keyboard_mode_active
                            last_mode_toggle_time = current_time
                            last_action_time = (
                                current_time  # Prevent immediate click after toggle
                            )
                            print(
                                f"Keyboard Mode {'ACTIVATED' if keyboard_mode_active else 'DEACTIVATED'}"
                            )
                            typed_text_preview = ""  # Clear preview on mode switch
                            pressed_key_char = None  # Clear highlight

                    if SHOW_WEBCAM_FEED:
                        cv2.circle(frame_bgr, wrist_px, 10, fist_color, cv2.FILLED)

                    # --- Mouse Clicks (Only if Keyboard Mode is OFF and not a fist) ---
                    if not keyboard_mode_active and not is_fist:
                        left_click_dist = calculate_distance(index_px, thumb_px)
                        right_click_dist = calculate_distance(middle_px, thumb_px)

                        # Left Click Check
                        left_click_color = (0, 255, 0)
                        if left_click_dist < LEFT_CLICK_DISTANCE_THRESHOLD:
                            left_click_color = (0, 0, 255)
                            if current_time - last_action_time > ACTION_COOLDOWN:
                                print(
                                    f"Action: Left Click! Dist: {left_click_dist:.2f}"
                                )
                                pyautogui.click(button="left")
                                last_action_time = current_time

                        # Right Click Check
                        right_click_color = (0, 255, 0)
                        if right_click_dist < RIGHT_CLICK_DISTANCE_THRESHOLD:
                            right_click_color = (0, 0, 255)
                            if current_time - last_action_time > ACTION_COOLDOWN:
                                print(
                                    f"Action: Right Click! Dist: {right_click_dist:.2f}"
                                )
                                pyautogui.click(button="right")
                                last_action_time = current_time

                        # Draw click indicators
                        if SHOW_WEBCAM_FEED:
                            cv2.line(frame_bgr, index_px, thumb_px, left_click_color, 2)
                            cv2.circle(
                                frame_bgr, index_px, 7, left_click_color, cv2.FILLED
                            )
                            cv2.line(
                                frame_bgr, middle_px, thumb_px, right_click_color, 2
                            )
                            cv2.circle(
                                frame_bgr, middle_px, 7, right_click_color, cv2.FILLED
                            )
                            thumb_action_color = (
                                (0, 0, 255)
                                if left_click_color == (0, 0, 255)
                                or right_click_color == (0, 0, 255)
                                else (0, 255, 0)
                            )
                            cv2.circle(
                                frame_bgr, thumb_px, 7, thumb_action_color, cv2.FILLED
                            )
                else:
                    if SHOW_WEBCAM_FEED:
                        cv2.putText(
                            frame_bgr,
                            "Left Hand Partially Out",
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 165, 255),
                            2,
                        )

            except Exception as e:
                print(f"Error processing Left Hand: {e}")
                # traceback.print_exc() # Uncomment for detailed errors

        # --- Process Right Hand (Mouse Movement OR Visual Keyboard Input) ---
        if right_hand_landmarks:
            try:
                index_tip_right = right_hand_landmarks.landmark[
                    mp_hands.HandLandmark.INDEX_FINGER_TIP
                ]
                fingertip_coords = get_landmark_pixel(
                    index_tip_right, frame_width, frame_height
                )

                if fingertip_coords:  # Ensure index tip is valid
                    # --- Visual Keyboard Input (Only if Keyboard Mode is ON) ---
                    if keyboard_mode_active:
                        key_pressed_in_frame = False
                        if (
                            not visual_key_cooldown_active
                        ):  # Only check if cooldown expired
                            pressed_key_char = (
                                None  # Reset highlight unless a press happens now
                            )
                            for key in keyboard_layout:
                                kx, ky, kw, kh = key["x"], key["y"], key["w"], key["h"]
                                char = key["char"]

                                # Check if fingertip is inside the key boundaries
                                if (
                                    kx < fingertip_coords[0] < kx + kw
                                    and ky < fingertip_coords[1] < ky + kh
                                ):
                                    key_to_press = char.lower()  # Default to lowercase
                                    if char == "<-":
                                        key_to_press = "backspace"
                                    elif char == "SPACE":
                                        key_to_press = "space"
                                    elif len(char) > 1:
                                        key_to_press = None  # Ignore multi-char keys for now unless handled

                                    if key_to_press:
                                        print(
                                            f"Visual KB Press: {char} ({key_to_press})"
                                        )
                                        pyautogui.press(key_to_press)
                                        last_key_press_time = (
                                            current_time  # Start cooldown
                                        )
                                        pressed_key_char = char  # For highlighting
                                        # Update preview text
                                        if key_to_press == "backspace":
                                            typed_text_preview = typed_text_preview[:-1]
                                        elif key_to_press == "space":
                                            typed_text_preview += " "
                                        else:
                                            typed_text_preview += char  # Add original char (case sensitive) to preview
                                        key_pressed_in_frame = True
                                        break  # Process only one key per frame check

                        # Draw pointer for visual keyboard
                        if SHOW_WEBCAM_FEED:
                            pointer_color = (
                                (0, 0, 255)
                                if visual_key_cooldown_active
                                else (255, 0, 0)
                            )  # Red during cooldown, Blue otherwise
                            cv2.circle(
                                frame_bgr,
                                fingertip_coords,
                                10,
                                pointer_color,
                                cv2.FILLED,
                            )

                    # --- Mouse Movement (Only if Keyboard Mode is OFF) ---
                    elif not keyboard_mode_active:
                        norm_x = index_tip_right.x
                        norm_y = index_tip_right.y

                        # Map hand position to screen position with edge reduction
                        target_screen_x = np.interp(
                            norm_x,
                            [FRAME_REDUCTION_X, 1.0 - FRAME_REDUCTION_X],
                            [0, SCREEN_WIDTH],
                        )
                        target_screen_y = np.interp(
                            norm_y,
                            [FRAME_REDUCTION_Y, 1.0 - FRAME_REDUCTION_Y],
                            [0, SCREEN_HEIGHT],
                        )

                        # Apply smoothing
                        current_x = prev_x + (target_screen_x - prev_x) / SMOOTHING
                        current_y = prev_y + (target_screen_y - prev_y) / SMOOTHING

                        # Clamp to screen bounds and move mouse
                        mouse_x = max(0, min(int(current_x), SCREEN_WIDTH - 1))
                        mouse_y = max(0, min(int(current_y), SCREEN_HEIGHT - 1))
                        pyautogui.moveTo(
                            mouse_x, mouse_y, duration=0
                        )  # duration=0 for instant move

                        prev_x, prev_y = current_x, current_y

                        # Draw movement indicator
                        if SHOW_WEBCAM_FEED:
                            cv2.circle(
                                frame_bgr, fingertip_coords, 12, (0, 255, 255), 3
                            )  # Yellow movement circle
                else:
                    if SHOW_WEBCAM_FEED:
                        cv2.putText(
                            frame_bgr,
                            "Right Hand Partially Out",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 165, 255),
                            2,
                        )

            except Exception as e:
                print(f"Error processing Right Hand: {e}")
                # traceback.print_exc() # Uncomment for detailed errors

        # --- Display Mode Status & Typed Text Preview ---
        if SHOW_WEBCAM_FEED:
            # Mode Status
            mode_text = "MODE: KEYBOARD" if keyboard_mode_active else "MODE: MOUSE"
            mode_color = (
                (0, 255, 255) if keyboard_mode_active else (255, 255, 0)
            )  # Yellow / Cyan
            cv2.putText(
                frame_bgr,
                mode_text,
                (10, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                mode_color,
                2,
            )

            # Typed Text Preview (Below Mode Status)
            preview_label = "Preview: "
            cv2.putText(
                frame_bgr,
                preview_label + typed_text_preview,
                (10, frame_height - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            # Optional: Add a background rect for preview text
            # text_size_preview = cv2.getTextSize(preview_label + typed_text_preview, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            # cv2.rectangle(frame_bgr, (8, frame_height - 50 - text_size_preview[1] - 2), (12 + text_size_preview[0], frame_height - 50 + 5), (50, 50, 50), -1)
            # cv2.putText(frame_bgr, preview_label + typed_text_preview, (10, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- Display Frame ---
        if SHOW_WEBCAM_FEED:
            cv2.imshow("Virtual OS Control", frame_bgr)

        # --- Quit Condition ---
        key = cv2.waitKey(5) & 0xFF
        if key == ord("q"):
            print("Quit signal received.")
            break
        # Allow closing window to quit
        if (
            SHOW_WEBCAM_FEED
            and cv2.getWindowProperty("Virtual OS Control", cv2.WND_PROP_VISIBLE) < 1
        ):
            print("Window closed manually.")
            break

except KeyboardInterrupt:
    print("\nKeyboard Interrupt received. Exiting.")
finally:
    # --- Cleanup ---
    cap.release()
    if SHOW_WEBCAM_FEED:
        cv2.destroyAllWindows()
    hands_detector.close()
    print("Resources released.")
