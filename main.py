import requests
import logging
import json
import random
import time
import uuid
import tempfile
import os
import google.generativeai as genai
from openai import OpenAI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a request model
class AudioRequest(BaseModel):
    audio_url: str

gemini_api_keys = [
    "AIzaSyCzmcLIlYR0kUsrZmTHolm_qO8yzPaaUNk",
    "AIzaSyDxxzuuGGh1wT_Hjl7-WFNDXR8FL72XeFM",
    "AIzaSyCGcxNi_ToOOWXGIKmByzJOAdRldAwiAvo",
    "AIzaSyCVTbP_VjBEYRU1OFWGoSbaGXIZN8KNeXY",
    "AIzaSyBaJWXjRAd39VYzGCmoz-yv4tJ6FiNTvIs"
]

# Function to select a random API key
def get_random_api_key():
    return random.choice(gemini_api_keys)

# Function to create generation configuration for keyword generation
def create_generation_config():
    return {
        "temperature": 2,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

# Function to generate keywords from text chunks
def process_chunk(chunk):
    max_attempts = len(gemini_api_keys)
    delay = 2  # Start with a 2-second delay
    attempts = 0

    while attempts < max_attempts:
        api_key = get_random_api_key()
        genai.configure(api_key=api_key)

        try:
            model = genai.GenerativeModel(
                model_name="gemini-1.5-pro-002",
                generation_config=create_generation_config(),
                system_instruction="""# Instructions

Given the following video script and captions, extract three visually concrete and specific keywords from each sentence that can be used to search for background videos. The keywords should be short (preferably 1-2 words) and capture the main essence of the sentence. If a keyword is a single word, return another visually concrete keyword related to it. The list must always contain the most relevant and appropriate query searches.

For example, if the caption is 'The cheetah is the fastest land animal, capable of running at speeds up to 75 mph', the keywords should include 'cheetah', 'speed', and 'running'. Similarly, for 'The Great Wall of China is one of the most iconic landmarks in the world', the keywords should be 'Great Wall', 'landmark', and 'China'.

Please return the keywords in a simple format, without numbering or any additional prefixes, such as:
- mountain peak
- challenging trail
- difficult journey
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
                delay *= 2  # Exponential backoff
            else:
                break

    logging.error("All API keys exhausted or failed.")
    return ""

# Function to download audio file using temporary files
def download_audio(audio_url):
    try:
        response = requests.get(audio_url)
        response.raise_for_status()  # Raise an error for bad responses
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(response.content)
            return tmp_file.name
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download audio: {e}")
        return None

# Function to handle transcription and fix the error
def create_json_response(transcription):
    try:
        # Extract transcription segments directly (no .get() needed)
        segments = transcription.segments if transcription else []
        response_data = {"segments": [{"start": segment.start, "end": segment.end, "text": segment.text} for segment in segments]}
        return json.dumps(response_data)
    except Exception as e:
        logging.error(f"Error creating JSON response: {e}")
        return {}

# FastAPI route to process the audio
@app.post("/process-audio/")
async def process_audio(request: AudioRequest):
    audio_url = request.audio_url
    audio_file = download_audio(audio_url)

    if audio_file is None:
        return {"error": "Failed to download audio"}

    # Transcription step (assuming you've integrated transcription API here)
    transcription = some_transcription_function(audio_file)  # You would implement or replace this

    # Convert transcription to JSON response
    json_response = create_json_response(transcription)
    return json_response
