import cv2
from fer import FER
import time

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 is usually the default value for the primary camera

# Initialize the FER detector
detector = FER()

start_time = time.time()
last_detection_time = start_time

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    current_time = time.time()
    elapsed_time = current_time - start_time

    # Perform emotion detection every second
    if current_time - last_detection_time >= 1:
        last_detection_time = current_time
        emotions = detector.detect_emotions(frame)
        for face in emotions:
            top_emotion, score = max(face['emotions'].items(), key=lambda item: item[1])
            print(f"Time: {elapsed_time:.2f}s, Top Emotion: {top_emotion}, Score: {score}")

    # Break the loop after 60 seconds
    if elapsed_time > 60:
        break

    # Press Q to exit before 60 seconds
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
