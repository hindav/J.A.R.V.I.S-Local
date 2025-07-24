import subprocess
import time
import requests
import sys
import os

# Add root path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Gui')))
from Gui.GUI import JarvisTerminal

# Function to check if Ollama is already running
def is_ollama_running():
    try:
        response = requests.get("http://localhost:11434")
        return response.status_code == 200
    except:
        return False

# Function to start Ollama server
def start_ollama():
    print("Starting Ollama server...")
    # If ollama is in PATH
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Wait for Ollama to be ready
    for _ in range(20):
        if is_ollama_running():
            print("✅ Ollama server is running.")
            return
        time.sleep(1)
    print("❌ Failed to start Ollama server. Exiting.")
    sys.exit(1)

def main():
    if not is_ollama_running():
        start_ollama()
    else:
        print("✅ Ollama server is already running.")

    print("Launching JARVIS...")
    app = JarvisTerminal()
    app.mainloop()

if __name__ == "__main__":
    main()
