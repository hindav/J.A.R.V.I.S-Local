import pyttsx3
import re

def clean_text(text):
    # Remove emojis and non-verbal symbols, but keep readable punctuation
    return re.sub(r'[^\w\s,.!?]', '', text)

def Speak(Text):
    engine = pyttsx3.init("sapi5")
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 170)

    cleaned = clean_text(Text)

    print("")
    print(f"Jarvis : {cleaned}")
    print("")
    engine.say(cleaned)
    engine.runAndWait()