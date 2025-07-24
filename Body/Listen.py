import speech_recognition as sr
import os

# --- Helper Functions ---

def read_mic(file_path):
    """Read the mic name from a file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Mic config file not found: {file_path}")
    mic_name = open(file_path).read().strip()
    if not mic_name:
        raise ValueError("Mic.txt is empty. Please specify a microphone name.")
    return mic_name

def read_keywords(file_path):
    """Read keywords from a file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Keywords file not found: {file_path}")
    with open(file_path, 'r') as file:
        return [line.strip().lower() for line in file.readlines() if line.strip()]

# --- Configuration ---

DEDICATED_MIC_NAME = read_mic('Data/Mic.txt')
KEYWORDS = read_keywords('Data/Keywords.txt')

# --- Get Microphone Index ---

def get_dedicated_microphone_index():
    mic_list = sr.Microphone.list_microphone_names()
    for index, mic_name in enumerate(mic_list):
        if DEDICATED_MIC_NAME.lower() in mic_name.lower():
            return index
    raise Exception(f"Dedicated microphone '{DEDICATED_MIC_NAME}' not found.")

# --- Listen for a Keyword Trigger ---

def ListenForKeyword():
    mic_index = get_dedicated_microphone_index()
    r = sr.Recognizer()

    with sr.Microphone(device_index=mic_index) as source:
        r.adjust_for_ambient_noise(source)
        print("🎙️ Listening for keywords...")

        while True:
            try:
                audio = r.listen(source)
                recognized_text = r.recognize_google(audio, language="en").lower()
                print(f"Recognized: {recognized_text}")

                for keyword in KEYWORDS:
                    if keyword in recognized_text:
                        print(f"✅ Keyword '{keyword}' detected!")
                        return keyword
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"Google API Error: {e}")
                return ""

# --- Listen for the Actual Command ---

def ListenForCommands():
    mic_index = get_dedicated_microphone_index()
    r = sr.Recognizer()

    with sr.Microphone(device_index=mic_index) as source:
        r.adjust_for_ambient_noise(source)
        print("🎧 Listening for command...")

        try:
            audio = r.listen(source)
            command = r.recognize_google(audio, language="en")
            print(f"🗣️ Command received: {command}")
            return command
        except sr.UnknownValueError:
            print("Could not understand audio.")
            return ""
        except sr.RequestError as e:
            print(f"Google API Error: {e}")
            return ""

# --- Combine Both for Final Call ---

def MicExecution():
    try:
        detected_keyword = ListenForKeyword()
        if detected_keyword:
            return ListenForCommands()
    except Exception as e:
        print(f"[MicExecution Error] {e}")
        return ""

# Uncomment below for testing standalone
# if __name__ == "__main__":
#     print("Listening started...")
#     command = MicExecution()
#     print("Final Command:", command)
