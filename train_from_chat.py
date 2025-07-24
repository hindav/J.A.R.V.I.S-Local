import json
import os
import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

# Ensure punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem_word(word):
    return stemmer.stem(word.lower())

def clean_sentence(sentence):
    return [stem_word(w) for w in tokenize(sentence) if w.isalnum()]

# Load existing tasks
with open("Data/Tasks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Load chat log
if not os.path.exists("chat_log.txt"):
    print("✅ No chat log found. Skipping chat training.")
    exit()

with open("chat_log.txt", "r", encoding="utf-8") as f:
    chat = f.read().strip()

# Simple Q&A extraction
lines = chat.splitlines()
pairs = []
for i in range(len(lines) - 1):
    if lines[i].startswith("You: ") and lines[i + 1].startswith("Jarvis: "):
        question = lines[i][4:].strip()
        answer = lines[i + 1][7:].strip()
        if question and answer:
            pairs.append((question, answer))

# Add to intents
if not pairs:
    print("✅ No new valid Q&A pairs found in chat log.")
    exit()

for q, a in pairs:
    # Look for an existing "chat" tag, else create one
    chat_intent = next((intent for intent in data["intents"] if intent["tag"] == "chat"), None)
    if chat_intent:
        chat_intent["patterns"].append(q)
        chat_intent["responses"].append(a)
    else:
        new_intent = {
            "tag": "chat",
            "patterns": [q],
            "responses": [a]
        }
        data["intents"].append(new_intent)

# Save updated tasks
with open("Data/Tasks.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"✅ Added {len(pairs)} Q&A pairs from chat log to Tasks.json.")
