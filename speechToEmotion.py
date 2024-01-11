import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import speech_recognition as sr
import nltk
from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# 安装nltk的punkt和vader_lexicon包
nltk.download('punkt')
nltk.download('vader_lexicon')


def record_audio(duration, filename='full_recording.wav'):
    fs = 44100  # frequency
    channels = 2  # number of channels

    # Recoding
    print("Start recoding...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='float64')
    sd.wait()
    print("Recoding ends")

    # Normalize floating point numbers and convert to 16-bit PCM format
    recording_normalized = np.int16(recording / np.max(np.abs(recording)) * 32767)

    # Save Recording
    write(filename, fs, recording_normalized)


def analyze_emotion_from_audio(filename, segment_duration, language='en-US',
                               output_filename='sentences_and_emotions.txt'):
    # Sentiment Analyzer
    sia = SentimentIntensityAnalyzer()
    recognizer = sr.Recognizer()

    with sr.AudioFile(filename) as source:
        total_duration = source.DURATION
        num_segments = int(np.ceil(total_duration / segment_duration))

        for i in range(num_segments):
            # Calculate the start and end time of each paragraph
            start = i * segment_duration
            end = min(start + segment_duration, total_duration)
            audio_data = recognizer.record(source, duration=end - start, offset=start)

            try:
                text = recognizer.recognize_google(audio_data, language=language)
                print(f"paragraph {i + 1} 's converted texts'：", text)
            except sr.UnknownValueError:
                print(f"Cannot recognize paragraph {i + 1} 's speech'")
                continue

            sentences = sent_tokenize(text)

            with open(output_filename, 'a') as file:
                file.write(f"Paragraph {i + 1} 's analysis result':\n")
                for sentence in sentences:
                    sentiment = sia.polarity_scores(sentence)
                    file.write(f"Sentence: '{sentence}'\n")
                    file.write(f"Sentiment: {sentiment}\n")
                file.write("\n")

    print(f"all sentiment analysis result save to '{output_filename}'")



record_audio(60, 'full_recording.wav')

analyze_emotion_from_audio('full_recording.wav', 30)
