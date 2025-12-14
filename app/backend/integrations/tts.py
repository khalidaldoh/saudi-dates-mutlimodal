from elevenlabs.client import ElevenLabs
from elevenlabs import save
from dotenv import load_dotenv
import os
import uuid
load_dotenv()

client = ElevenLabs(
    api_key=os.getenv("ELEVEN_LABS_TTS_API")
)

# Generates TTS audio from text and returns the path to the saved audio file

def generate_tts(text: str) -> str:
    # ... (existing client setup) ...

    # 1. Define the directory
    output_dir = "static/images/audio"

    # 2. Create it if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{uuid.uuid4()}.mp3"
    # 3. Use os.path.join for safety
    file_path = os.path.join(output_dir, filename)

    audio = client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    save(audio, file_path)
    return file_path

