import tkinter as tk
import threading
import time
import os
import sys

# Add root path to import Main.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Main import MainTaskExecution
from Body.Listen import MicExecution
from Body.Speak import Speak
from Brain.AIBrain import ReplyBrain

boot_lines = [
    "[ JARVIS v1.0 Bootloader ]",
    "Initializing core modules...",
    "Loading AI kernel...",
    "Establishing secure connection...",
    "Calibrating voice interface...",
    "System boot complete.",
    "Developed by @hindav"
]

class JarvisTerminal(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("JARVIS Terminal")
        self.state('zoomed')
        self.configure(bg="#000000")

        self.output = tk.Text(self, bg="#000000", fg="#00FF00", font=("Consolas", 14),
                              insertbackground="#00FF00", wrap=tk.WORD, padx=10, pady=10, bd=0)
        self.output.pack(fill=tk.BOTH, expand=True)
        self.output.config(state=tk.DISABLED)

        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(self, textvariable=self.input_var, bg="#000000", fg="#00FF00",
                                    font=("Consolas", 14), insertbackground="#00FF00", relief=tk.FLAT, bd=0)
        self.input_entry.bind("<Return>", self.process_command)

        self.protocol("WM_DELETE_WINDOW", self.close)
        self.quit_prompt_shown = False
        self.chat_mode = False
        self.speech_enabled = True  # New flag to control speech output

        self.bind("<Escape>", self.ask_quit)
        self.bind("q", self.confirm_quit)

        # Create button frame for better layout
        self.button_frame = tk.Frame(self, bg="#000000")
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Chat mode toggle button
        self.toggle_button = tk.Button(self.button_frame, text="Switch to Chat Mode", bg="#222222", fg="white",
                                       font=("Consolas", 12), command=self.toggle_chat_mode, relief=tk.FLAT)
        self.toggle_button.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Speech toggle button
        self.speech_button = tk.Button(self.button_frame, text="🔊 Speech ON", bg="#222222", fg="white",
                                       font=("Consolas", 12), command=self.toggle_speech, relief=tk.FLAT)
        self.speech_button.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        threading.Thread(target=self.boot_sequence, daemon=True).start()

    def boot_sequence(self):
        name = "Boss"
        try:
            with open("Name.txt", "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    name = content
        except FileNotFoundError:
            pass

        for line in boot_lines:
            self.append_output(line)
            time.sleep(1)

        self.append_output(f">>> Welcome back, {name}.")
        self.after(0, self.show_input)
        self.after(0, self.start_voice_loop)

        # Test speech during boot
        if self.speech_enabled:
            self.speak_async(f"Welcome back, {name}. All systems online.")

    def show_input(self):
        self.append_output("\nType your command below or use voice commands.")
        if self.chat_mode:
            self.input_entry.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
            self.input_entry.focus()

    def append_output(self, text):
        def update_text():
            self.output.config(state=tk.NORMAL)
            self.output.insert(tk.END, text + "\n")
            self.output.see(tk.END)
            self.output.config(state=tk.DISABLED)
        
        if threading.current_thread() != threading.main_thread():
            self.after(0, update_text)
        else:
            update_text()

    def speak_async(self, text):
        """Async speech with better error handling"""
        if not self.speech_enabled:
            print(f"🔇 Speech disabled - would say: {text}")
            return
            
        def speak_thread():
            try:
                print(f"🗣️ Speaking: {text}")
                Speak(text)
                print(f"✅ Speech completed: {text}")
            except Exception as e:
                print(f"❌ Speech error: {e}")
                self.append_output(f"Speech error: {e}")
        
        threading.Thread(target=speak_thread, daemon=True).start()

    def process_command(self, event):
        query = self.input_var.get().strip()
        self.input_var.set("")

        if not query:
            return

        if query.lower() in ["exit", "quit"]:
            self.append_output(">>> Exiting JARVIS...")
            if self.speech_enabled:
                self.speak_async("Goodbye. JARVIS shutting down.")
            self.after(1500, self.close)
            return

        self.append_output(f"You: {query}")
        
        # Process command in thread but ensure speech works
        threading.Thread(target=self.execute_task, args=(query, "chat")).start()

    def execute_task(self, query, source="voice"):
        """Execute task with proper source tracking"""
        try:
            print(f"🎯 Executing task: '{query}' from {source}")
            
            # Try main task execution first
            response = MainTaskExecution(query)
            
            if response:
                self.append_output(f"JARVIS: {response}")
                if self.speech_enabled:
                    print(f"🔊 Task response - speaking: {response}")
                    self.speak_async(response)
            else:
                # Fall back to AI brain
                print(f"🧠 No task match, using AI brain for: {query}")
                reply = ReplyBrain(query)
                self.append_output(f"JARVIS: {reply}")
                if self.speech_enabled:
                    print(f"🔊 AI response - speaking: {reply}")
                    self.speak_async(reply)
                    
        except Exception as e:
            error_msg = f"Error processing command: {e}"
            print(f"❌ {error_msg}")
            self.append_output(f"JARVIS: {error_msg}")
            if self.speech_enabled:
                self.speak_async("I encountered an error processing your request.")

    def start_voice_loop(self):
        threading.Thread(target=self.voice_loop, daemon=True).start()

    def voice_loop(self):
        """Voice loop that respects chat mode"""
        while True:
            try:
                # Skip voice input in chat mode
                if self.chat_mode:
                    time.sleep(2)
                    continue
                
                print("🎤 Listening for voice input...")
                data = MicExecution()
                data = str(data).strip()
                
                if len(data) <= 3:
                    continue
                    
                print(f"🎤 Voice input received: {data}")
                self.append_output(f"You (Voice): {data}")
                
                # Execute voice command
                self.execute_task(data, "voice")
                
            except Exception as e:
                error_msg = f"Mic error: {e}"
                print(f"❌ {error_msg}")
                self.append_output(error_msg)
                time.sleep(2)  # Wait before retrying

    def toggle_chat_mode(self):
        """Toggle between chat and voice mode"""
        self.chat_mode = not self.chat_mode
        
        if self.chat_mode:
            self.input_entry.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
            self.input_entry.focus()
            self.toggle_button.config(text="Switch to Voice Mode")
            self.append_output(">>> Chat mode activated - Type your commands")
            if self.speech_enabled:
                self.speak_async("Chat mode activated")
        else:
            self.input_entry.pack_forget()
            self.toggle_button.config(text="Switch to Chat Mode")
            self.append_output(">>> Voice mode activated - Speak your commands")
            if self.speech_enabled:
                self.speak_async("Voice mode activated")

    def toggle_speech(self):
        """Toggle speech output on/off"""
        self.speech_enabled = not self.speech_enabled
        
        if self.speech_enabled:
            self.speech_button.config(text="🔊 Speech ON", fg="white")
            self.append_output(">>> Speech output enabled")
            self.speak_async("Speech output enabled")
        else:
            self.speech_button.config(text="🔇 Speech OFF", fg="gray")
            self.append_output(">>> Speech output disabled")

    def ask_quit(self, event):
        if not self.quit_prompt_shown:
            self.quit_prompt_shown = True
            self.append_output("\n>>> Press 'Q' to confirm quit.")
            if self.speech_enabled:
                self.speak_async("Press Q to quit")

    def confirm_quit(self, event):
        if self.quit_prompt_shown:
            self.append_output(">>> Quitting JARVIS...")
            if self.speech_enabled:
                self.speak_async("Goodbye. JARVIS shutting down.")
            self.after(1500, self.close)

    def close(self):
        print("🔴 JARVIS Terminal closing...")
        self.destroy()


if __name__ == "__main__":
    print("🚀 Starting JARVIS Terminal...")
    app = JarvisTerminal()
    app.mainloop()