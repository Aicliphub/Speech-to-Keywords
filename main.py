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

# Bulk process function to generate keywords for multiple chunks
def process_chunks_in_bulk(chunks):
    max_attempts = len(gemini_api_keys)
    delay = 2  # Start with a 2-second delay
    attempts = 0
    bulk_results = []

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

            # Send chunks as a bulk request
            response = model.start_chat(history=[{"role": "user", "parts": chunks}]).send_message(" ".join(chunks))
            
            if response and hasattr(response, 'text') and response.text:
                bulk_results.extend(response.text.strip().splitlines())
                break  # Successfully processed the bulk request, exit the loop

        except Exception as e:
            logging.error(f"API key {api_key} failed: {e}")

            if "Resource has been exhausted" in str(e) or "429" in str(e):
                attempts += 1
                logging.info(f"Retrying with a new API key after {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                break

    if not bulk_results:
        logging.error("All API keys exhausted or failed.")
    
    # Ensure bulk_results has results for each chunk, fill in blank keywords if necessary
    while len(bulk_results) < len(chunks):
        bulk_results.append("")

    return bulk_results

# Function to download audio file using temporary files
def download_audio(audio_url):
    try:
        response = requests.get(audio_url)
        response.raise_for_status()  # Raise an error for bad responses
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
            audio_file.write(response.content)
            audio_filename = audio_file.name  # Get the name of the temporary file

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
                response_format="verbose_json"  # Requesting verbose JSON format
            )
            logging.info(f"Transcription Response: {transcript}")
            return transcript
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return None

# Function to extract segments from the transcription response
def extract_segments(transcription):
    return transcription.segments if hasattr(transcription, 'segments') else []

# Function to create JSON response from transcription
def create_json_response(transcription):
    lines_with_keywords = []

    # Extract segments using the new function
    segments = extract_segments(transcription)
    total_segments = len(segments)

    # Process chunks in bulk
    chunk_texts = [segment.text for segment in segments]
    keywords_bulk = process_chunks_in_bulk(chunk_texts)

    for i in range(total_segments):
        segment = segments[i]
        text = segment.text if hasattr(segment, 'text') else ''
        start_time = segment.start if hasattr(segment, 'start') else 0
        finish_time = segments[i + 1].start if i < total_segments - 1 else segment.end if hasattr(segment, 'end') else start_time + 1

        lines_with_keywords.append({
            "text": text,
            "keyword": keywords_bulk[i],
            "start": start_time,
            "finish": finish_time
        })

    return {"transcription": lines_with_keywords}

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
