import os
import random
import logging
import time
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# FastAPI app initialization
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for the incoming request
class JsonUrlRequest(BaseModel):
    json_url: str

# List of Gemini API keys
gemini_api_keys = [
    "AIzaSyCzmcLIlYR0kUsrZmTHolm_qO8yzPaaUNk",
    "AIzaSyDxxzuuGGh1wT_Hjl7-WFNDXR8FL72XeFM",
    "AIzaSyCGcxNi_ToOOWXGIKmByzJOAdRldAwiAvo",
    "AIzaSyCVTbP_VjBEYRU1OFWGoSbaGXIZN8KNeXY",
    "AIzaSyBaJWXjRAd39VYzGCmoz-yv4tJ6FiNTvIs"
]

# Function to get a random API key
def get_random_api_key():
    """Select a random API key from the list."""
    return random.choice(gemini_api_keys)

# Function to create generation config for Gemini API
def create_generation_config():
    """Create the model configuration."""
    return {
        "temperature": 0.7,  # Lower temperature for more focused output
        "top_p": 0.95,
        "top_k": 50,
        "max_output_tokens": 150,  # Limit to shorter response
        "response_mime_type": "text/plain",
    }

# Function to process chunk and generate keywords
def process_chunk(chunk):
    """Generate keywords from the given chunk using the Gemini API with retries and key rotation."""
    max_attempts = 3
    delay = 2  # Start with a 2-second delay
    attempts = 0

    while attempts < max_attempts:
        api_key = get_random_api_key()
        genai.configure(api_key=api_key)

        try:
            # Create the model
            model = genai.GenerativeModel(
                model_name="gemini-1.5-pro-002",
                generation_config=create_generation_config(),
                system_instruction="Generate a few short keywords for the given scene description that could help in searching for an image."
            )

            # Start a chat session with the chunk of text
            chat_session = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [chunk],
                    },
                ]
            )

            # Log the response status for debugging
            logging.info(f"API Key: {api_key}, Attempt: {attempts + 1}")

            # Check if response exists
            if hasattr(chat_session, 'response') and chat_session.response:
                response = chat_session.response
                logging.info(f"Response received: {response}")
                return response
            else:
                logging.error("No response from Gemini API")
        except Exception as e:
            logging.error(f"Attempt {attempts + 1} failed with error: {str(e)}")
        
        attempts += 1
        time.sleep(delay)

    raise HTTPException(status_code=500, detail="Failed to generate keywords after multiple attempts.")

@app.post("/generate_keywords")
async def generate_keywords(request: JsonUrlRequest):
    """Receive a JSON URL, process it and generate keywords for each scene."""
    try:
        # Fetch the JSON data from the provided URL
        response = requests.get(request.json_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch JSON data from the URL")

        json_data = response.json()
        logging.info(f"Successfully fetched JSON data: {json_data}")

        # Extract scene text chunks from the JSON data
        scene_chunks = [scene.get('text', '') for scene in json_data if 'text' in scene]

        # Generate keywords for each scene chunk
        result = []
        for chunk in scene_chunks:
            if chunk.strip():  # Only process non-empty chunks
                keywords = process_chunk(chunk)
                result.append({
                    "text": chunk,
                    "keywords": keywords,
                })

        return result

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
