import os
import requests
import logging
import json
import random
import time
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# API keys for Google Generative API (Gemini) and OpenAI Whisper API
gemini_api_keys = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
    os.getenv("GEMINI_API_KEY_5")
]
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI
app = FastAPI()

# Request model for the API
class AudioURLRequest(BaseModel):
    audio_url: str

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

# Function to download audio file
def download_audio(audio_url):
    audio_filename = f"audio_{uuid.uuid4()}.mp3"  # Create a unique filename
    try:
        response = requests.get(audio_url)
        response.raise_for_status()  # Raise an error for bad responses
        with open(audio_filename, 'wb') as audio_file:
            audio_file.write(response.content)
        logging.info(f"Downloaded audio file: {audio_filename}")
    except Exception as e:
        logging.error(f"Error downloading audio file: {e}")
        return None
    return audio_filename

# Function to transcribe audio
def transcribe_audio(audio_filename):
    client = OpenAI(
        api_key=openai_api_key,
        base_url="https://api.lemonfox.ai/v1"
    )
    
    try:
        with open(audio_filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en",
                response_format="verbose_json"  # Requesting verbose JSON format
            )
            return transcript
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return None

# Function to create JSON response from transcription
def create_json_response(transcription):
    lines_with_keywords = []  # List to hold lines with their keywords

    for segment in transcription['segments']:  # Access segments correctly
        text = segment.text  # Access the 'text' attribute of the segment object
        keyword = process_chunk(text)  # Generate keyword for the scene
        
        # Append to the list for JSON response without timing
        lines_with_keywords.append({"text": text, "keyword": keyword})

    return lines_with_keywords

# Clean up temporary files after processing
def cleanup_file(filepath):
    try:
        os.remove(filepath)
        logging.info(f"Deleted file: {filepath}")
    except Exception as e:
        logging.error(f"Failed to delete file {filepath}: {e}")

# FastAPI endpoint for processing audio URL
@app.post("/transcribe_and_generate_keywords")
async def transcribe_and_generate_keywords(request: AudioURLRequest, background_tasks: BackgroundTasks):
    audio_filename = download_audio(request.audio_url)
    
    if not audio_filename:
        raise HTTPException(status_code=400, detail="Failed to download audio.")

    # Step 2: Transcribe the audio
    transcription = transcribe_audio(audio_filename)
    
    if not transcription:
        raise HTTPException(status_code=500, detail="Failed to transcribe audio.")

    # Generate JSON response with keywords
    json_response = create_json_response(transcription)

    # Schedule file cleanup after response
    background_tasks.add_task(cleanup_file, audio_filename)
    
    return json_response
