from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
import os

load_dotenv()

print("KEY LOADED :", os.getenv("SARVAM_API_KEY"))
print("CWD", os.getcwd())

source = "https://youtu.be/SCkP1UOYaCw?si=8CGWmHKUFe38Tr95"
language = "hinglish"  # change to "Hinglish" to test Sarvam

chunks = process_input(source)
transcript = transcribe_all(chunks, language=language)

print("\n=== TRANSCRIPT ===\n")
print(transcript)