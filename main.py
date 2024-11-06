import requests
import logging
import random
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

# Function to generate keywords from the transcription text file using Gemini
def generate_keywords_from_textfile(transcription_text):
    api_key = get_random_api_key()
    genai.configure(api_key=api_key)
    
    # Write transcription to a temp text file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as text_file:
        text_file.write(transcription_text.encode('utf-8'))
        text_filename = text_file.name

    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-002",
            generation_config={
                "temperature": 2,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            },
            system_instruction="""# Instructions
            
Extract three visually concrete and specific keywords from each line for background video searches. 
Return each line's keywords in a simple format:
- cheetah speed running
- Great Wall landmark China
"""
        )
        
        # Read the transcription text from the temp file
        with open(text_filename, "r") as file:
            transcription_content = file.read()

        chat_session = model.start_chat(history=[{"role": "user", "parts": [transcription_content]}])
        response = chat_session.send_message(transcription_content)
        
        if response and hasattr(response, 'text') and response.text:
            keywords = response.text.strip().split("\n")
        else:
            keywords = []
        
    except Exception as e:
        logging.error(f"Keyword generation error: {e}")
        keywords = []

    os.remove(text_filename)  # Clean up temp file
    return keywords

# FastAPI endpoint for audio processing
@app.post("/process-audio/")
async def process_audio(request: AudioRequest):
    audio_url = request.audio_url
    audio_filename = download_audio(audio_url)
    
    if audio_filename:
        transcription = transcribe_audio(audio_filename)
        
        if transcription:
            # Join all segment texts to create a single transcription text
            transcription_text = "\n".join(segment['text'] for segment in transcription.segments)
            keywords = generate_keywords_from_textfile(transcription_text)

            # Remove audio file after processing
            os.remove(audio_filename)

            # Return JSON response containing only keywords for each segment
            return {"keywords": keywords}
        else:
            return {"error": "Transcription failed"}
    else:
        return {"error": "Audio download failed"}

# Main script
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
