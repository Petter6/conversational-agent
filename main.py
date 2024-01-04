import whisper
import spacy


# Whisper takes an audio file
def transcription(audio):
    transcribe = model.transcribe(audio, fp16=False)
    return transcribe["text"]


# spaCy takes the string and outputs the sentences
def indexing(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    for sent in doc.sents:
        print(sent)


if __name__ == '__main__':
    model = whisper.load_model("base")

    audio = "landslide.mp3"

    text = transcription(audio)

    indexing(text)
