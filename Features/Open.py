import os
import subprocess
import webbrowser
import re
import win32com.client  # pip install pywin32
from Body.Speak import Speak

def Open(command):
    command = command.lower().replace("open ", "").strip()

    # 1. Check if it's a URL or domain
    if re.match(r"(https?://|www\.|[a-zA-Z0-9\-]+\.[a-zA-Z]{2,})", command):
        if not command.startswith("http"):
            command = "https://" + command
        webbrowser.open(command)
        Speak(f"Opening {command}")
        return

    # 2. Known quick-access apps (optional)
    quick_apps = {
        "chrome": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        "vlc": "C:\\Program Files\\VideoLAN\\VLC\\vlc.exe",
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "whatsapp": os.path.expanduser("~\\AppData\\Local\\WhatsApp\\WhatsApp.exe"),
    }

    for name, path in quick_apps.items():
        if name in command:
            try:
                subprocess.Popen(path)
                Speak(f"Opening {name}")
                return
            except Exception:
                pass  # fallback to below

    # 3. Search Start Menu Shortcuts (.lnk)
    start_menu_paths = [
        os.path.join(os.environ['APPDATA'], r'Microsoft\Windows\Start Menu\Programs'),
        r'C:\ProgramData\Microsoft\Windows\Start Menu\Programs'
    ]

    for path in start_menu_paths:
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".lnk") and command in file.lower():
                    shortcut_path = os.path.join(root, file)
                    try:
                        shell = win32com.client.Dispatch("WScript.Shell")
                        shortcut = shell.CreateShortCut(shortcut_path)
                        target_path = shortcut.Targetpath
                        subprocess.Popen(target_path)
                        Speak(f"Opening {file.replace('.lnk', '')}")
                        return
                    except Exception:
                        Speak("I found it but couldn’t open it.")
                        return

    # 4. Scan Program Files for .exe
    program_dirs = [
        "C:\\Program Files",
        "C:\\Program Files (x86)",
        os.path.expanduser("~\\AppData\\Local\\Programs"),
    ]

    for base in program_dirs:
        for root, dirs, files in os.walk(base):
            for file in files:
                if file.endswith(".exe") and command in file.lower():
                    exe_path = os.path.join(root, file)
                    try:
                        subprocess.Popen(exe_path)
                        Speak(f"Opening {file}")
                        return
                    except Exception:
                        continue

    Speak("I couldn't find or open that application or website.")
