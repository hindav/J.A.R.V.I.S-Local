import nltk
from nltk.stem.porter import PorterStemmer
import torch
import torch.nn as nn
import json
import numpy as np
import random
from Features.Open import Open  # Ensure this exists and handles logic safely

# Load punkt tokenizer if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Stemmer and Tokenization Helpers
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

# Load model and training data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('Data/Tasks.json', 'r') as f:
    intents = json.load(f)

FILE = "Data/Tasks.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# Neural Net Model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        return self.l3(x)

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Main Inference Logic
def preprocess_query(query):
    tokens = tokenize(query)
    return bag_of_words(tokens, all_words)

def TasksExecutor(X):
    X = torch.from_numpy(X).to(device).unsqueeze(0)  # Shape: (1, input_size)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    prob = torch.softmax(output, dim=1)[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    return None

def MainTaskExecution(query):
    task = query.lower()
    X = preprocess_query(task)
    result = TasksExecutor(X)

    try:
        if result and ("open" in result or "start" in result):
            Open(task)  # Delegate app/site launching
            return "✅ Task executed to open application or website."
    except Exception as e:
        print(f"[❌ Open Error]: {e}")

    return result

# For quick testing
# def MainExecution():
#     sample_query = "open Whatsapp"
#     result = MainTaskExecution(sample_query)
#     if result:
#         print(f"🎯 Task Executed: {result}")
#     else:
#         print("🤖 No valid task found.")


# if __name__ == "__main__":
#     MainExecution()
