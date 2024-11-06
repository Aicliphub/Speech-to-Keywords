from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import httpx
import random
import logging
import google.generativeai as genai

# Initialize FastAPI app
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Gemini API keys
gemini_api_keys = [
    "AIzaSyCzmcLIlYR0kUsrZmTHolm_qO8yzPaaUNk",
    "AIzaSyDxxzuuGGh1wT_Hjl7-WFNDXR8FL72XeFM",
    "AIzaSyCGcxNi_ToOOWXGIKmByzJOAdRldAwiAvo",
    "AIzaSyCVTbP_VjBEYRU1OFWGoSbaGXIZN8KNeXY",
    "AIzaSyBaJWXjRAd39VYzGCmoz-yv4tJ6FiNTvIs"
]

def get_random_api_key():
    """Select a random API key from the list."""
    return random.choice(gemini_api_keys)

def create_generation_config():
    """Create the model configuration."""
    return {
        "temperature": 2,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 100,
        "response_mime_type": "text/plain",
    }

async def fetch_json(url: str):
    """Fetch JSON data from a given URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()

async def generate_keywords_for_text(text):
    """Generate keywords using the Gemini API with retries and API key rotation."""
    max_attempts = len(gemini_api_keys)
    attempts = 0

    while attempts < max_attempts:
        api_key = get_random_api_key()
        genai.configure(api_key=api_key)

        try:
            # Create the model
            model = genai.GenerativeModel(
                model_name="gemini-1.5-pro-002",
                generation_config=create_generation_config(),
                system_instruction="""Extract three visually concrete and specific keywords from each line for background video searches. Return each line's keywords in a simple format."""
            )

            # Start chat session with the text to generate keywords
            chat_session = model.start_chat(
                history=[{"role": "user", "parts": [text]}]
            )
            response = chat_session.response
            return response

        except Exception as e:
            logging.error(f"Attempt {attempts + 1} failed: {e}")
            attempts += 1

    raise HTTPException(status_code=500, detail="Failed to generate keywords after multiple attempts.")

# Define the request schema
class JsonUrlRequest(BaseModel):
    json_url: str

@app.post("/generate_keywords")
async def generate_keywords(request: JsonUrlRequest):
    try:
        # Fetch JSON data from the provided URL
        data = await fetch_json(request.json_url)

        # Generate keywords for each text segment
        results = []
        for item in data:
            text = item["text"]
            keywords = await generate_keywords_for_text(text)
            item_with_keywords = {
                "start": item["start"],
                "end": item["end"],
                "text": item["text"],
                "text_offset": item["text_offset"],
                "keywords": keywords
            }
            results.append(item_with_keywords)

        return results

    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP error fetching JSON: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch JSON data.")
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while generating keywords.")
