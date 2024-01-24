import concurrent.futures

from furhat_remote_api import FurhatRemoteAPI
from spacy.tokens import Doc
from transformers import pipeline
import cv2
from fer import FER
import spacy
import numpy as np
import tempfile
import pyaudio
import wave
import whisper
import threading
import os
import time
from textblob import TextBlob
import json
import pycountry
import numerizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Decides when to stop listening
# Change this value depending on your environment
# For me 50 - 70 worked best
silence_threshold = 70

# Load the Whisper model once
model = whisper.load_model("base.en")

# Load the spaCy model once
nlp = spacy.load("en_core_web_md")

# Flag to signal when to stop capturing video
capture_video_flag = threading.Event()


def analyze_emotion(text):
    # Using a pre-trained model for emotion recognition
    classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")

    results = classifier(text, return_all_scores=True)

    return results


def listen():
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav")

    sample_rate = 16000
    bits_per_sample = 16
    chunk_size = 1024
    audio_format = pyaudio.paInt16
    channels = 1

    # Parameters for detecting silence
    silence_duration_threshold = 2  # 2 seconds of silence
    silence_duration = 0

    # Event to signal the end of recording
    end_recording_event = threading.Event()

    def callback(in_data, frame_count, time_info, status):
        nonlocal silence_duration

        wav_file.writeframes(in_data)
        audio_data = np.frombuffer(in_data, dtype=np.int16)

        # Check for silence
        rms = np.sqrt(np.mean(np.maximum(audio_data ** 2, 0)))  # Ensure non-negative values
        if not np.isnan(rms) and rms < silence_threshold:
            silence_duration += chunk_size / sample_rate
        else:
            silence_duration = 0

        # If silence duration exceeds the threshold, set the end_recording_event
        if silence_duration >= silence_duration_threshold:
            end_recording_event.set()

        return None, pyaudio.paContinue

    # Open the wave file for writing
    wav_file = wave.open(temp_file.name, 'wb')
    wav_file.setnchannels(channels)
    wav_file.setsampwidth(bits_per_sample // 8)
    wav_file.setframerate(sample_rate)

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Start recording audio
    stream = audio.open(format=audio_format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size,
                        stream_callback=callback)

    try:
        # Wait until end_recording_event is set or Ctrl+C is pressed
        end_recording_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        # Stop and close the audio stream
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Close the wave file
        wav_file.close()

        # Transcribe the audio to text (suppressing warnings about running on a CPU)
        result = model.transcribe(temp_file.name, fp16=False)
        temp_file.close()

        return result["text"].strip()


def video_thread():
    cap = cv2.VideoCapture(0)  # 0 is usually the default value for the primary camera
    detector = FER()

    # Lists to store emotion scores for each frame
    emotion_scores_list = []

    while not capture_video_flag.is_set():
        ret, frame = cap.read()

        if not ret:
            break

        emotions = detector.detect_emotions(frame)
        if emotions is not None and emotions:
            # Process and use the 'emotions' data as needed
            emotion_scores_list.append(list(emotions[0]['emotions'].values()))

        # Wait for the next photo interval
        time.sleep(1)

    # When capture_video_flag is set, release the capture
    cap.release()
    cv2.destroyAllWindows()

    # Compute the average emotion scores
    if emotion_scores_list:
        average_emotions = np.mean(emotion_scores_list, axis=0)
        print("Average Emotions:", average_emotions)
        return average_emotions
    else:
        print("No frames captured.")
        return False


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

        'joy': 3,
        'sadness': 4,
        'anger': 0,
        'fear': 2,
        'surprise': 5,
    }

    # Calculating the combined emotional scores
    emotions_array = {}
    text_weight = 0.8  # Weight for text analysis results
    video_weight = 0.2  # Weight for video analysis results

    for emotion, text_score in text_emotions.items():
        if emotion in emotion_mapping:
            video_score = video_emotions[emotion_mapping[emotion]]
            combined_score = text_score * text_weight + video_score * video_weight
        else:
            # If the emotion is not present in the video analysis, use only the text score
            combined_score = text_score * text_weight

        emotions_array[emotion] = combined_score
    return emotions_array


def get_answer():
    text = listen()

    while not text or text == '.':
        furhat.say(
            text="Sorry I couldn't understand, could you repeat?", blocking=True)
        text = listen()

    return text


def get_country(text):
    for country in pycountry.countries:
        if country.name in text:
            return country.name
    return ""


def get_answer_and_emotion():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Start the video thread
        video_future = executor.submit(video_thread)

        # Run the listen function in the main thread
        result1 = listen()

        # Signal the video thread to stop
        capture_video_flag.set()

        # Get the result of the video thread
        result2 = video_future.result()

    text_emotion = analyze_emotion(result1)
    video_emotion = result2

    # Parsing
    text_emotions_parsed = parse_text_emotions(text_emotion)

    combined_emotions = get_combined_score(text_emotions_parsed, video_emotion)

    capture_video_flag.clear()

    return {"message": result1, "emotion": combined_emotions}


def is_answer_positive(answer):
    # Create a TextBlob object
    blob = TextBlob(answer)

    # Get the polarity score
    polarity = blob.sentiment.polarity

    # Define a threshold for considering a response as positive using polarity
    polarity_threshold = 0.2

    # Check if the polarity is above the positive threshold
    if polarity > polarity_threshold:
        return True

    # Check for positive keywords in the response
    positive_keywords = ['yes', 'sure', 'absolutely', 'agree', 'like', 'correct', 'yeah', 'right']

    for word in positive_keywords:
        if word in answer.lower():
            return True

    return False


def check_for(text, pattern):
    doc = nlp(text)

    for token in doc:
        if token.text.lower() in pattern:
            return str(token)

    return False


def get_locations_date(location_type):
    result = get_answer_and_emotion()
    doc = nlp(result["message"])

    country_name = get_country(result["message"])

    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]

    while country_name == "":
        furhat.say(text=f"What country did you go to?", blocking=True)
        message = get_answer()
        country_name = get_country(message)

    while not dates:
        furhat.say(text=f"Can you repeat your date", blocking=True)
        doc = nlp(get_answer())
        dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]

    # Create a dictionary for each entry with separate emotion keys
    entry = {"type": location_type, "country": country_name, "date": dates}
    entry.update(result["emotion"])

    return entry


def get_number():
    doc = nlp(get_answer())

    numbers = doc._.numerize()

    while not numbers:
        furhat.say(text=f"Can you repeat your duration", blocking=True)
        doc = nlp(get_answer())
        numbers = doc._.numerize()

    # Return the first value of the dictionary
    first_number = next(iter(numbers.values()), None)

    return first_number


if __name__ == '__main__':
    # Create an instance of the FurhatRemoteAPI class
    furhat = FurhatRemoteAPI("localhost")

    # Data that will be written to the json file
    data = []
    data_trips = []

    # Ask for the user's name
    furhat.say(text="Hi there, my name is Matthew, I'm going to help you find your next holiday destination. What is "
                    "your name?", blocking=True)
    name = get_answer()

    # Ask for the user's country
    furhat.say(text="What country do you currently reside in?", blocking=True)
    country = ""
    while country == "":
        country = get_country(get_answer())

    # Ask for the user's country
    furhat.say(text="What city do you currently live in?", blocking=True)
    city = get_answer()

    # Ask for the user's country
    furhat.say(text="For how many weeks do you want to go on vacation?", blocking=True)
    duration = get_number()
    print(duration)

    # Ask if it is the users dream trip
    furhat.say(text=f"Hi {name}, Is this your dream trip, so money is not an issue?", blocking=True)
    dream = get_answer()

    if is_answer_positive(dream):
        standard_of_living = 'dream'
        standard_of_holiday = 'dream'
    else:
        # Ask for the user's standard of living
        furhat.say(text=f"How would you describe your current standard of living at your current location.",
                   blocking=True)

        standard_of_living = False

        while not standard_of_living:
            furhat.say(text="Are you a low, medium or high spender in your city.", blocking=True)
            answer = get_answer()
            standard_of_living = check_for(answer, ["low", "medium", "high"])

        # Ask for the user's standard of living whilst on holiday
        furhat.say(text="What is your desired cost of living at your holiday destination?", blocking=True)

        standard_of_holiday = False

        while not standard_of_holiday:
            furhat.say(text="Will you be a low medium or high spender at the destination?", blocking=True)
            answer = get_answer()
            standard_of_holiday = check_for(answer, ["low", "medium", "high"])

    # Ask the user if they have ever been to a beach town
    furhat.say(text=f"Have you ever been on holiday to a town near a beach?", blocking=True)
    beach = get_answer()

    while is_answer_positive(beach):
        furhat.say(text=f"Could you tell me about a time you went to a town near a beach?", blocking=True)
        response = get_locations_date("beach")
        data_trips.append(response)
        furhat.say(text=f"Were there any other times you went on a holiday to a town near a beach?", blocking=True)
        beach = get_answer()
    
    # Ask the user if they have ever been to a cultural town
    furhat.say(text=f"Have you ever been on holiday to a cultural town?", blocking=True)
    cultural = get_answer()

    while is_answer_positive(cultural):
        furhat.say(text=f"Could you tell me about a time you went to a cultural town?", blocking=True)
        response = get_locations_date("cultural")
        data_trips.append(response)
        furhat.say(text=f"Were there any other times you went on a holiday to a cultural town?", blocking=True)
        cultural = get_answer()

    # Ask the user if they have ever been to a festival town
    furhat.say(text=f"Have you ever been on holiday to a festival town?", blocking=True)
    festival = get_answer()

    while is_answer_positive(festival):
        furhat.say(text=f"Could you tell me about a time you went to a festival town?", blocking=True)
        response = get_locations_date("festival")
        data_trips.append(response)
        furhat.say(text=f"Were there any other times you went on a holiday to a festival town?", blocking=True)
        festival = get_answer()

    # Ask the user if they have ever been to a nightlife town
    furhat.say(text=f"Have you ever been on holiday to a nightlife town?", blocking=True)
    nightlife = get_answer()

    while is_answer_positive(nightlife):
        furhat.say(text=f"Could you tell me about a time you went to a nightlife town?", blocking=True)
        response = get_locations_date("nightlife")
        data_trips.append(response)
        furhat.say(text=f"Were there any other times you went on a holiday to a nightlife town?", blocking=True)
        nightlife = get_answer()

    # Ask the user if they have ever been to a mountain town
    furhat.say(text=f"Have you ever been on holiday to a mountain town?", blocking=True)
    mountain = get_answer()

    while is_answer_positive(mountain):
        furhat.say(text=f"Could you tell me about a time you went to a mountain town?", blocking=True)
        response = get_locations_date("mountain")
        data_trips.append(response)
        furhat.say(text=f"Were there any other times you went on a holiday to a mountain town?", blocking=True)
        mountain = get_answer()

    entry = {"name": name, "country": country, "city": city, "standard_of_living": standard_of_living,
             "standard_of_holiday": standard_of_holiday, "duration": duration, "trips": data_trips}
