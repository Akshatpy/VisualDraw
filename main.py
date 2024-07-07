import cv2
import mediapipe as mp
import numpy as np
import ctypes
import time
def get_screen_resolution(): #To make the program functional for different screen sizes
    user32 = ctypes.windll.user32
    screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return screensize

def drawlines(canvas, points, color=(255, 255, 255), thickness=3):
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        cv2.line(canvas, points[i - 1], points[i], color, thickness)
def draw_text(canvas, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2):
    cv2.putText(canvas, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
def run():
    capture = cv2.VideoCapture(0)  # OpenCV Webcam VideoCapture object
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    screen_width, screen_height = get_screen_resolution()
    speed = 0.45  # basically the speed
    ball_x, ball_y = None, None  # whatever initial ball location
    drawing_points = []
    cv2.namedWindow('Visual Drawer', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Visual Drawer', screen_width, screen_height)
    cv2.setWindowProperty('Visual Drawer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    start_time = time.time()
    drawing_text = True
    started_time = time.time()
    display_text = True
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                target_x = int(index_finger_tip.x * frame.shape[1])
                target_y = int(index_finger_tip.y * frame.shape[0])
                if ball_x is None or ball_y is None:
                    ball_x, ball_y = target_x, target_y
                else:
                    ball_x = int(ball_x * (1 - speed) + target_x * speed)
                    ball_y = int(ball_y * (1 - speed) + target_y * speed)
        else:
            ball_x, ball_y = None, None
        if ball_x is not None and ball_y is not None:
            ball_center = (ball_x, ball_y)
            ball_radius = 20
            canvas = np.zeros_like(frame, dtype=np.uint8)
            cv2.circle(canvas, ball_center, ball_radius, (0, 255, 0), -1)
            drawing_points.append(ball_center)
        canvas = np.zeros_like(frame, dtype=np.uint8)
        drawlines(canvas, drawing_points)
        if display_text and time.time() - start_time < 5:
            draw_text(canvas, "Move fingertip and press q to exit", (50, screen_height - 50))
        else:
            display_text = False
        cv2.imshow('Visual Drawer', canvas)
        if cv2.waitKey(1) & 0xFF == ord('c') or cv2.waitKey(1) & 0xFF == ord('C'):
            drawing_points = []
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('Q'):
            break
    capture.release()
    cv2.destroyAllWindows()
    print("Running main script...")

if __name__ == "__main__":
    run()

