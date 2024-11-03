import os
import logging
import tempfile
import httpx
import random
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize FastAPI app
app = FastAPI()

# Load API keys from environment variables
gemini_api_keys = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
    os.getenv("GEMINI_API_KEY_5")
]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

# Define request model
class AudioRequest(BaseModel):
    audio_url: str

# Function to generate keywords from text chunks
async def process_chunk(chunk):
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

            response = await chat_session.send_message(chunk)
            if response and hasattr(response, 'text') and response.text:
                return response.text.strip()

        except Exception as e:
            logging.error(f"API key {api_key} failed: {e}")

            if "Resource has been exhausted" in str(e) or "429" in str(e):
                attempts += 1
                logging.info(f"Retrying with a new API key after {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                break

    logging.error("All API keys exhausted or failed.")
    return ""

# Function to download audio file
async def download_audio(audio_url):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(audio_url)
            response.raise_for_status()  # Raise an error for bad responses
            return response.content  # Return the audio content
    except Exception as e:
        logging.error(f"Error downloading audio file: {e}")
        return None

# Function to transcribe audio
async def transcribe_audio(audio_filename):
    client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.lemonfox.ai/v1")
    
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

    for segment in transcription['segments']:
        text = segment['text']
        # Use asyncio.run to call process_chunk
        keyword = asyncio.run(process_chunk(text))  # Generate keyword for the scene
        
        # Append to the list for JSON response
        lines_with_keywords.append({"text": text, "keyword": keyword})

    return {"segments": lines_with_keywords}

# Endpoint for transcribing audio and generating keywords
@app.post("/transcribe/")
async def transcribe_audio_endpoint(request: AudioRequest, background_tasks: BackgroundTasks):
    try:
        # Step 1: Download the audio file
        audio_content = await download_audio(request.audio_url)

        if audio_content:
            # Step 2: Use a temporary file to save the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
                temp_audio_file.write(audio_content)
                temp_audio_filename = temp_audio_file.name
            
            # Step 3: Transcribe the audio
            transcription = await transcribe_audio(temp_audio_filename)

            # Cleanup the temporary file
            os.remove(temp_audio_filename)

            if transcription:
                # Create JSON response
                json_response = create_json_response(transcription)
                return JSONResponse(content=json_response)

        raise HTTPException(status_code=500, detail="Error processing the audio file.")
    
    except Exception as e:
        logging.error(f"Error in transcribe_audio_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Health check endpoint
@app.get("/health/")
async def health_check():
    return {"status": "ok"}

# Main execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
