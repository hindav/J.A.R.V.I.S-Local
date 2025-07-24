import speech_recognition as sr

print("Available microphones:")
for i, mic_name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"Index {i}: {mic_name}")