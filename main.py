import concurrent.futures

from furhat_remote_api import FurhatRemoteAPI
from transformers import pipeline
import cv2
from fer import FER
import spacy
import numerizer


def analyze_emotion(text):
    # Using a pre-trained model for emotion recognition
    classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")

    results = classifier(text, return_all_scores=True)

    return results


def listen():
    text = furhat.listen()
    return text.message


def video():
    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 is usually the default value for the primary camera

    # Initialize the FER detector
    detector = FER()

    ret, frame = cap.read()

    if not ret:
        return

    emotions = detector.detect_emotions(frame)

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


def get_combined_score(text_emotions, video_emotions):
    emotion_mapping = {
        'joy': 'happy',
        'sadness': 'sad',
        'anger': 'angry',
        'fear': 'fear',
        'surprise': 'surprise'
    }

    # Calculating the combined emotional scores
    emotions_array = {}
    text_weight = 0.8  # Weight for text analysis results
    video_weight = 0.2  # Weight for video analysis results

    for emotion, text_score in text_emotions.items():
        if emotion in emotion_mapping:
            video_score = emotion_mapping[emotion]
            combined_score = text_score * text_weight + video_emotions.get(video_score, 0) * video_weight
        else:
            # If the emotion is not present in the video analysis, use only the text score
            combined_score = text_score * text_weight

        emotions_array[emotion] = combined_score

    return emotions_array


def get_answer():
    text = furhat.listen()

    while not text.message:
        furhat.say(
            text="Sorry I couldn't understand, could you repeat?", blocking=True)
        text = furhat.listen()

    return text.message


def get_answer_and_emotion():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(listen)
        future2 = executor.submit(video)

        result1 = future1.result()
        result2 = future2.result()

    text_emotion = analyze_emotion(result1)
    video_emotion = result2

    # Parsing
    text_emotions_parsed = parse_text_emotions(text_emotion)
    video_emotions_parsed = parse_video_emotions(video_emotion)

    combined_emotions = get_combined_score(text_emotions_parsed, video_emotions_parsed)

    return result1, combined_emotions


def get_budget():
    doc = nlp(get_answer())
    positive_patterns = ['yes', 'sure', 'absolutely', 'agree', 'like', 'correct', 'yeah']

    budget_items = list(doc._.numerize().items())

    while not budget_items:
        furhat.say(text=f"Can you repeat your budget", blocking=True)
        doc = nlp(get_answer())
        budget_items = list(doc._.numerize().items())

    budget = budget_items[0][1]
    furhat.say(text=f"So your budget is {budget}?", blocking=True)

    while True:
        doc = nlp(get_answer())
        for token in doc:
            if token.text.lower() in positive_patterns:
                return budget

        while not doc.has_extension("numerize"):
            furhat.say(text=f"Can you repeat your budget", blocking=True)
            doc = nlp(get_answer())

        budget_items = list(doc._.numerize().items())

        while not budget_items:
            furhat.say(text=f"Can you repeat your budget", blocking=True)
            doc = nlp(get_answer())
            budget_items = list(doc._.numerize().items())

        budget = budget_items[0][1]
        furhat.say(text=f"So your budget is {budget}?", blocking=True)


if __name__ == '__main__':
    nlp = spacy.load("en_core_web_md")

    # Create an instance of the FurhatRemoteAPI class
    furhat = FurhatRemoteAPI("localhost")

    furhat.say(text="Hi there, my name is Matthew, I'm going to help you find your next vacation destination. What is "
                    "your name?", blocking=True)
    name = get_answer()

    furhat.say(text=f"Hi {name}, what is your budget?", blocking=True)
    total_budget = get_budget()

    furhat.say(text=f"have you been to any beach towns when did you go and how did you find it", blocking=True)
    response = get_answer_and_emotion()
    print(response)
