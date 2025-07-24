import os
import sys
import torch
import random
import json
import requests
import re
import subprocess
from nltk.stem.porter import PorterStemmer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from Features.Open import Open
except ImportError:
    def Open(x): pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("🔧 Updating model with chat_log.txt ...")
subprocess.run([sys.executable, "train_from_chat.py"])
print("✅ Model update complete.\n")

with open('Data/Tasks.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

FILE = "Data/Tasks.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

import torch.nn as nn
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        return self.l3(x)

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

stemmer = PorterStemmer()

def tokenize(sentence):
    return re.findall(r"\b\w+\b", sentence.lower())

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = [1 if w in sentence_words else 0 for w in words]
    return torch.tensor(bag, dtype=torch.float32)

MEMORY_PATH = "Data/memory.json"

def load_memory():
    if os.path.exists(MEMORY_PATH):
        try:
            with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except json.JSONDecodeError:
            print("⚠️ Warning: memory.json is corrupted. Resetting memory.")
            return {}
    return {}


def save_memory(memory):
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=4)

def extract_key_value(sentence):
    if " is " in sentence:
        parts = sentence.split(" is ", 1)
        key = parts[0].lower().strip()
        value = parts[1].strip()
        return key, value
    return None

def get_local_reply(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words).unsqueeze(0).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                
                # Special logic for "hello" tag to defer complex sentences to GPT
                if tag == "hello":
                    basic_keywords = intent["patterns"]
                    msg_lower = msg.lower().strip()

                    # If msg is longer than a basic greeting, let GPT handle
                    if not any(msg_lower == kw.lower() for kw in basic_keywords):
                        return None

                reply = random.choice(intent['responses'])

                if "open" in reply.lower() or "start" in reply.lower():
                    Open(msg)
                return reply
    return None

def get_gpt_reply(prompt):
    try:
        with open("Data/Api.txt", "r") as file:
            url = file.read().strip()
    except FileNotFoundError:
        return "❌ Error: Data/Api.txt not found."

    payload = {
        "model": "llama3:latest",
        "prompt": prompt,
        "stream": True
    }

    try:
        response = requests.post(url, json=payload, stream=True, timeout=30)

        if response.status_code != 200:
            return "❌ Ollama stream error."

        full_reply = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                token = data.get("response", "")
                full_reply += token
                print(token, end='', flush=True)

        print()
        return full_reply.strip()

    except requests.exceptions.RequestException:
        return "❌ Could not reach Ollama server."

def ReplyBrain(query):
    memory = load_memory()

    if query.lower().startswith("remember "):
        fact = query[9:].strip()
        key_val = extract_key_value(fact)
        if key_val:
            key, value = key_val
            memory[key] = value
            save_memory(memory)
            return f"✅ Remembered: {key} is {value}"
        else:
            return "❌ Could not understand what to remember."

    for key, value in memory.items():
        if key in query.lower():
            return f"You told me {key} is {value}."

    local_reply = get_local_reply(query)
    if local_reply:
        return local_reply

    chat_path = "chat_log.txt"
    try:
        with open(chat_path, "r", encoding="utf-8") as f:
            chat_log = f.read()
    except FileNotFoundError:
        chat_log = ""

    prompt = f"{chat_log}\nYou: {query}\nJarvis:"
    reply = get_gpt_reply(prompt)

    with open(chat_path, "a", encoding="utf-8") as f:
        f.write(f"\nYou: {query}\nJarvis: {reply}\n")

    return reply

if __name__ == "__main__":
    print("🤖 Jarvis AI Brain running...\n")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Jarvis: Goodbye 👋")
                break
            response = ReplyBrain(user_input)
            print("Jarvis:", response)
        except KeyboardInterrupt:
            print("\nJarvis: Interrupted. See you later!")
            break
