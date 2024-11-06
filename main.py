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

# Function to download audio file using temporary files
def download_audio(audio_url):
    try:
        response = requests.get(audio_url)
        response.raise_for_status()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
            audio_file.write(response.content)
            audio_filename = audio_file.name

        logging.info(f"Downloaded audio file: {audio_filename}")
    except Exception as e:
        logging.error(f"Error downloading audio file: {e}")
        return None
    return audio_filename

# Function to transcribe audio
def transcribe_audio(audio_filename):
    client = OpenAI(
        api_key="FZqncRg9uxcpdKH4WVghkmtiesRr2S50",
        base_url="https://api.lemonfox.ai/v1"
    )
    
    try:
        with open(audio_filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en",
                response_format="verbose_json"
            )
            logging.info(f"Transcription Response: {transcript}")
            return transcript
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return None

# Function to process multiple segments for keyword generation in bulk
def process_segments_bulk(segments):
    text_chunks = [segment['text'] for segment in segments]
    chunk_text = "\n".join(text_chunks)
    
    api_key = get_random_api_key()
    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-002",
            generation_config=create_generation_config(),
            system_instruction="""# Instructions
            
Given the following video script and captions, extract three visually concrete and specific keywords from each line to use for background video searches. The keywords should be short and specific to the content of each line.

Please return the keywords in a simple format for each line, such as:
- cheetah speed running
- Great Wall landmark China
""",
        )
        
        chat_session = model.start_chat(history=[{"role": "user", "parts": [chunk_text]}])
        response = chat_session.send_message(chunk_text)
        keywords = response.text.strip().split("\n") if response and hasattr(response, 'text') and response.text else []
        
        # Map the keywords to segments
        for i, segment in enumerate(segments):
            segment["keyword"] = keywords[i] if i < len(keywords) else ""
        
    except Exception as e:
        logging.error(f"Keyword generation error: {e}")
        for segment in segments:
            segment["keyword"] = ""  # Leave empty if keyword generation fails

    return segments

# Function to create JSON response from transcription segments
def create_json_response(transcription):
    segments = [{"text": segment['text'], "start": segment['start'], "end": segment['end']} for segment in transcription.segments]
    segments = process_segments_bulk(segments)  # Process all segments at once for keywords
    
    return {"transcription": segments}

# FastAPI endpoint for audio processing
@app.post("/process-audio/")
async def process_audio(request: AudioRequest):
    audio_url = request.audio_url
    audio_filename = download_audio(audio_url)
    
    if audio_filename:
        transcription = transcribe_audio(audio_filename)
        
        if transcription:
            json_response = create_json_response(transcription)
            logging.info("Successfully processed audio.")
            os.remove(audio_filename)
            return json_response
        else:
            return {"error": "Transcription failed"}
    else:
        return {"error": "Audio download failed"}

# Main script
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
