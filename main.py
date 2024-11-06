import google.generativeai as genai
import requests
import logging
import json
import time
import tempfile
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import deque
from functools import wraps

from openai import OpenAI
from typing import Callable, List
from typing_extensions import TypedDict

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


# Function to download audio file using temporary files
def download_audio(audio_url):
    response = requests.get(audio_url)
    response.raise_for_status()  # Raise an error for bad responses

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
        audio_file.write(response.content)
        audio_filename = audio_file.name  # Get the name of the temporary file

    logging.info(f"Downloaded audio file: {audio_filename}")

    return audio_filename


# Function to transcribe audio
def transcribe_audio(audio_filename):
    client = OpenAI(
        api_key="FZqncRg9uxcpdKH4WVghkmtiesRr2S50",
        base_url="https://api.lemonfox.ai/v1",
    )

    with open(audio_filename, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en",
            response_format="verbose_json",  # Requesting verbose JSON format
        )
        logging.info(f"Transcription Response: {transcript}")
        return transcript


def create_generation_config():
    return genai.GenerationConfig(
        temperature=0.9,
        top_p=1,
        top_k=1,
        max_output_tokens=2048,
    )


class KeywordsResponse(TypedDict):
    keywords: List[List[str]]


class APIKeyManager:
    def __init__(self, keys):
        self.keys = deque([(key, 0) for key in keys])  # (key, last_used_time)

    def get_next_key(self):
        current_time = time.time()
        least_recently_used = min(self.keys, key=lambda x: x[1])
        self.keys.remove(least_recently_used)
        key, _ = least_recently_used
        self.keys.appendleft((key, current_time))
        return key


# Initialize the API key manager
gemini_api_keys = [
    "AIzaSyCzmcLIlYR0kUsrZmTHolm_qO8yzPaaUNk",
    "AIzaSyDxxzuuGGh1wT_Hjl7-WFNDXR8FL72XeFM",
    "AIzaSyCGcxNi_ToOOWXGIKmByzJOAdRldAwiAvo",
    "AIzaSyCVTbP_VjBEYRU1OFWGoSbaGXIZN8KNeXY",
    "AIzaSyBaJWXjRAd39VYzGCmoz-yv4tJ6FiNTvIs",
]
api_key_manager = APIKeyManager(gemini_api_keys)


def get_next_api_key():
    return api_key_manager.get_next_key()


def api_key_rotator(
    max_attempts: int = 5, initial_delay: float = 1, max_total_delay: float = 15
):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            total_delay = 0
            attempts = 0

            while attempts < max_attempts:
                api_key = get_next_api_key()
                genai.configure(api_key=api_key)
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logging.error(f"API key {api_key} failed: {e}")
                    if "Resource has been exhausted" in str(e) or "429" in str(e):
                        attempts += 1
                        delay = min(
                            initial_delay * (2 ** (attempts - 1)),
                            max_total_delay - total_delay,
                        )
                        total_delay += delay
                        logging.info(
                            f"Retrying with a new API key after {delay} seconds... (Total delay: {total_delay}s)"
                        )
                        time.sleep(delay)
                    else:
                        raise e

            raise Exception(f"All attempts failed after {max_attempts} tries")

        return wrapper

    return decorator


class Segment(BaseModel):
    text: str
    start: float
    finish: float
    keyword: str


@api_key_rotator()
def generate_keywords_from_segments(segments_json) -> List[Segment]:
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-002", generation_config=create_generation_config()
    )

    # Create a prompt for the Gemini model
    prompt = """
    Given the following video segments, extract three visually concrete and specific keywords from each segment that can be used to search for background videos. The keywords should be short (preferably 1-2 words) and capture the main essence of the segment. If a keyword is a single word, return another visually concrete keyword related to it. The list must always contain the most relevant and appropriate query searches.

    Please return the results as a list of lists, where each inner list contains three keywords for a segment.

    Here are the segments to process:
    """
    prompt += json.dumps([segment["text"] for segment in segments_json])
    prompt += f"\n\nMake sure to return exactly {len(segments_json)} lists of keywords, one for each segment."

    # Use the Gemini model to generate keywords
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json", response_schema=KeywordsResponse
        ),
    )

    if response and hasattr(response, "text") and response.text:
        keywords_response = json.loads(response.text)

        # Associate keywords with original segments
        segments_with_keywords = []
        for segment, keywords in zip(segments_json, keywords_response["keywords"]):
            segments_with_keywords.append(
                {
                    "text": segment["text"],
                    "keyword": "\n".join(keywords),
                    "start": segment["start"],
                    "finish": segment["finish"],
                }
            )

        return segments_with_keywords
    else:
        raise Exception("Empty or invalid response from Gemini API")


# FastAPI endpoint for audio processing
@app.post("/process-audio/")
async def process_audio(request: AudioRequest):
    audio_url = request.audio_url  # Access the audio_url from the request body
    # Step 1: Download the audio file
    try:
        audio_filename = download_audio(audio_url)
    except Exception as e:
        return {"error": f"An audio error happened {e}"}

    # Step 2: Transcribe the audio
    try:
        transcription = transcribe_audio(audio_filename)
    except Exception as e:
        return {"error": f"A transcription error happened {e}"}

    # Clean up the temporary audio file
    os.remove(audio_filename)  # Remove the temporary file

    # Ensure correct attributes are accessed
    segments_json = [
        {"text": segment.text, "start": segment.start, "finish": segment.finish}
        for segment in transcription.segments
    ]
    
    try:
        segments_with_keywords = generate_keywords_from_segments(segments_json)
    except Exception as e:
        return {"error": f"An LLM error happened {e}"}

    return {"transcriptions": segments_with_keywords}


# Main script
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
