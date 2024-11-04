from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import logging
import json
import random
import time
import uuid
import google.generativeai as genai
from openai import OpenAI
from tempfile import NamedTemporaryFile

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)

gemini_api_keys = [
    "AIzaSyCzmcLIlYR0kUsrZmTHolm_qO8yzPaaUNk",
    "AIzaSyDxxzuuGGh1wT_Hjl7-WFNDXR8FL72XeFM",
    "AIzaSyCGcxNi_ToOOWXGIKmByzJOAdRldAwiAvo",
    "AIzaSyCVTbP_VjBEYRU1OFWGoSbaGXIZN8KNeXY",
    "AIzaSyBaJWXjRAd39VYzGCmoz-yv4tJ6FiNTvIs"
]

class AudioRequest(BaseModel):
    audio_url: str

def get_random_api_key():
    return random.choice(gemini_api_keys)

def create_generation_config():
    return {
        "temperature": 2,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

def process_chunk(chunk):
    max_attempts = len(gemini_api_keys)
    delay = 2
    attempts = 0

    while attempts < max_attempts:
        api_key = get_random_api_key()
        genai.configure(api_key=api_key)

        try:
            model = genai.GenerativeModel(
                model_name="gemini-1.5-pro-002",
                generation_config=create_generation_config(),
                system_instruction="""# Instructions
                Given the following video script and captions, extract three visually concrete keywords...
                """
            )

            chat_session = model.start_chat(
                history=[{"role": "user", "parts": [chunk]}]
            )

            response = chat_session.send_message(chunk)
            if response and hasattr(response, 'text') and response.text:
                return response.text.strip()

        except Exception as e:
            logging.error(f"API key {api_key} failed: {e}")

            if "Resource has been exhausted" in str(e) or "429" in str(e):
                attempts += 1
                logging.info(f"Retrying with a new API key after {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                break

    logging.error("All API keys exhausted or failed.")
    return ""

def download_audio(audio_url):
    try:
        response = requests.get(audio_url)
        response.raise_for_status()
        temp_audio_file = NamedTemporaryFile(delete=True, suffix=".mp3")
        temp_audio_file.write(response.content)
        temp_audio_file.seek(0)
        logging.info("Audio downloaded and stored in temporary file.")
        return temp_audio_file
    except Exception as e:
        logging.error(f"Error downloading audio file: {e}")
        return None

def transcribe_audio(temp_audio_file):
    client = OpenAI(
        api_key="FZqncRg9uxcpdKH4WVghkmtiesRr2S50",
        base_url="https://api.lemonfox.ai/v1"
    )

    try:
        with temp_audio_file as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en",
                response_format="verbose_json"
            )
            return transcript
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return None

def create_json_response(transcription):
    lines_with_keywords = []

    for segment in transcription.segments:
        text = segment.text  # Correctly access the 'text' attribute
        keyword = process_chunk(text)
        lines_with_keywords.append({"text": text, "keyword": keyword})

    return lines_with_keywords

@app.post("/transcribe")
async def transcribe_audio_url(request: AudioRequest):
    audio_url = request.audio_url

    # Step 1: Download the audio file
    temp_audio_file = download_audio(audio_url)
    
    if temp_audio_file:
        # Step 2: Transcribe the audio
        transcription = transcribe_audio(temp_audio_file)
        
        if transcription:
            # Step 3: Create JSON response
            json_response = create_json_response(transcription)
            return json_response
        else:
            raise HTTPException(status_code=500, detail="Error during transcription.")
    else:
        raise HTTPException(status_code=400, detail="Failed to download audio file.")
