import concurrent.futures

from furhat_remote_api import FurhatRemoteAPI
from transformers import pipeline
import cv2
from fer import FER
import time


def analyze_emotion(text):
    # Using a pre-trained model for emotion recognition
    classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")

    results = classifier(text, return_all_scores=True)

    return results


def listen():
    text = furhat.listen()
    return text


def video():
    recording_time = 1
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
        # cv2.imshow('Webcam', frame)

        current_time = time.time()
        elapsed_time = current_time - start_time

        # Perform emotion detection every second
        if current_time - last_detection_time >= 1:
            last_detection_time = current_time
            emotions = detector.detect_emotions(frame)

        # Break the loop after the specified recording time
        if elapsed_time > recording_time:
            break

        # Press Q to exit before the specified recording time
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return emotions


def parse_text_emotions(data):
    """
    Parses the text emotion data format into a dictionary.

    Args:
    data (list): A list of dictionaries with 'label' and 'score' keys.

    Returns:
    dict: A dictionary with emotion labels as keys and their scores as values.
    """
    return {item['label']: item['score'] for item in data[0]}


def parse_video_emotions(data):
    """
    Parses the video emotion data format into a dictionary.

    Args:
    data (list): A list of dictionaries, where 'emotions' key contains a dictionary of emotions and scores.

    Returns:
    dict: A dictionary with emotion labels as keys and their scores as values.
    """
    # Assuming there's only one item in the list for video emotions
    return data[0]['emotions']


if __name__ == '__main__':
    # Create an instance of the FurhatRemoteAPI class
    furhat = FurhatRemoteAPI("localhost")

    # Say "Hi there, how are you?"
    furhat.say(text="Hi there, how are you?", blocking=True)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(listen)
        future2 = executor.submit(video)

        result1 = future1.result()
        result2 = future2.result()

    text_emotion = analyze_emotion(result1.message)
    video_emotion = result2

    # Parsing
    text_emotions_parsed = parse_text_emotions(text_emotion)
    video_emotions_parsed = parse_video_emotions(video_emotion)

    emotion_mapping = {
        'joy': 'happy',
        'sadness': 'sad',
        'anger': 'angry',
        'fear': 'fear',
        'surprise': 'surprise'
    }

    # Calculating the combined emotional scores
    combined_emotions = {}
    text_weight = 0.8  # Weight for text analysis results
    video_weight = 0.2  # Weight for video analysis results

    for text_emotion, text_score in text_emotions_parsed.items():
        if text_emotion in emotion_mapping:
            video_emotion = emotion_mapping[text_emotion]
            combined_score = text_score * text_weight + video_emotions_parsed.get(video_emotion, 0) * video_weight
        else:
            # If the emotion is not present in the video analysis, use only the text score
            combined_score = text_score * text_weight

        combined_emotions[text_emotion] = combined_score

    print(combined_emotions)
