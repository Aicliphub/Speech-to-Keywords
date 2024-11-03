import os
import requests
import logging
import random
import time
import uuid
import google.generativeai as genai
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the FastAPI app
app = FastAPI()

# Replace with your actual API keys from .env
gemini_api_keys = os.getenv("GEMINI_API_KEYS").split(",")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define the request body model
class AudioRequest(BaseModel):
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

# Function to generate keywords from text chunks with retries
def process_chunk(chunk):
    max_attempts = len(gemini_api_keys) + 3  # Allowing extra attempts for retry
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
                attempts += 1  # Increment attempts for other errors

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

    for segment in transcription['segments']:  # Access segments from the transcription response
        text = segment['text']
        keyword = process_chunk(text)  # Generate keyword for the scene
        start_time = segment['start']  # Assuming start time is available
        end_time = segment['end']  # Assuming end time is available
        
        # Append to the list for JSON response with timing
        lines_with_keywords.append({
            "text": text,
            "keyword": keyword,
            "start_time": start_time,
            "end_time": end_time
        })

    return lines_with_keywords

# Define the API endpoint for audio processing
@app.post("/process-audio")
async def process_audio(request: AudioRequest):
    audio_url = request.audio_url
    
    # Step 1: Download the audio file
    audio_filename = download_audio(audio_url)
    
    if audio_filename:
        try:
            # Step 2: Transcribe the audio
            transcription = transcribe_audio(audio_filename)
            
            if transcription:
                # Create JSON response
                json_response = create_json_response(transcription)
                return JSONResponse(content=json_response)
            else:
                raise HTTPException(status_code=500, detail="Transcription failed.")
        finally:
            # Clean up temporary audio file after processing
            try:
                os.remove(audio_filename)
                logging.info(f"Deleted temporary audio file: {audio_filename}")
            except Exception as e:
                logging.error(f"Error deleting audio file: {e}")
    else:
        raise HTTPException(status_code=500, detail="Audio download failed.")

# Run the application with the command: uvicorn your_filename:app --reload
