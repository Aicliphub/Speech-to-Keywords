import os
import random
import logging
import time
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests

# List of Gemini API keys
gemini_api_keys = [
    "AIzaSyCzmcLIlYR0kUsrZmTHolm_qO8yzPaaUNk",
    "AIzaSyDxxzuuGGh1wT_Hjl7-WFNDXR8FL72XeFM",
    "AIzaSyCGcxNi_ToOOWXGIKmByzJOAdRldAwiAvo",
    "AIzaSyCVTbP_VjBEYRU1OFWGoSbaGXIZN8KNeXY",
    "AIzaSyBaJWXjRAd39VYzGCmoz-yv4tJ6FiNTvIs"
]

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

# Function to get a random API key
def get_random_api_key():
    """Select a random API key from the list."""
    return random.choice(gemini_api_keys)

# Function to create the model configuration
def create_generation_config():
    """Create the model configuration."""
    return {
        "temperature": 2,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

# Function to process chunk and handle API errors
def process_chunk(chunk):
    """Generate keywords from the given chunk using the Gemini API with retries and API key rotation."""
    max_attempts = len(gemini_api_keys)
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
                system_instruction="""Generate a focused 2-3 word keyword spaced with comma, for each of the following scene lines that could be used to search for a suitable image on pexels.com. The keywords should be provided directly without any additional instructions or structural markers. Ensure to read all the lines and create an equivalent entity of keyword lines. Do not skip or forget a line."""
            )

            # Start a chat session with the provided chunk
            chat_session = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [chunk],
                    },
                ]
            )

            # Log the response status to better debug the issue
            logging.info(f"Chat session initiated with API key: {api_key}")

            # Check for response and log it for debugging
            if chat_session and hasattr(chat_session, 'response'):
                response = chat_session.response
                logging.info(f"Response received: {response}")
                return response  # Return the generated response
            else:
                logging.error("No response from Gemini API")
                attempts += 1
                time.sleep(delay)
        except Exception as e:
            logging.error(f"Attempt {attempts + 1} failed: {e}")
            attempts += 1
            time.sleep(delay)

    raise HTTPException(status_code=500, detail="Failed to generate keywords after multiple attempts.")

@app.post("/generate_keywords")
async def generate_keywords(request: JsonUrlRequest):
    """Receive a JSON URL and generate keywords from it."""
    try:
        # Fetch JSON data from the URL
        response = requests.get(request.json_url)
        if response.status_code == 200:
            json_data = response.json()
        else:
            raise HTTPException(status_code=400, detail="Failed to fetch JSON data from URL")

        # Extract scene text chunks from the JSON data
        scene_chunks = [scene['text'] for scene in json_data]

        # Generate keywords for each scene chunk
        result = []
        for chunk in scene_chunks:
            keywords = process_chunk(chunk)  # Get keywords for the scene
            result.append({
                "text": chunk,
                "keywords": keywords,
            })

        return result

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
