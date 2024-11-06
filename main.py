import asyncio
from asyncio import Queue
import aiohttp
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


# Shared queue and rate limiter
request_queue = Queue()
MAX_REQUESTS_PER_MINUTE = 60
RATE_LIMIT_PERIOD = 60  # seconds

# Asynchronous function to process requests from the queue
async def process_queue():
    async with aiohttp.ClientSession() as session:
        while True:
            chunk, future = await request_queue.get()
            try:
                result = await process_chunk_async(chunk, session)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                request_queue.task_done()
            
            # Rate limiting
            await asyncio.sleep(RATE_LIMIT_PERIOD / MAX_REQUESTS_PER_MINUTE)

# Asynchronous version of process_chunk
async def process_chunk_async(chunk, session):
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
                ... (your prompt here)
                """
            )
            chat_session = model.start_chat(
                history=[{"role": "user", "parts": [chunk]}]
            )
            response = await asyncio.to_thread(chat_session.send_message, chunk)
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

# Modified process_chunk function to use the queue
async def process_chunk(chunk):
    future = asyncio.Future()
    await request_queue.put((chunk, future))
    return await future

# ... rest of the existing code ...

# Modified create_json_response function to use async process_chunk
async def create_json_response(transcription):
    lines_with_keywords = []
    segments = extract_segments(transcription)
    total_segments = len(segments)
    
    # Process chunks concurrently
    tasks = []
    for i in range(total_segments):
        segment = segments[i]
        text = segment.text if hasattr(segment, 'text') else ''
        start_time = segment.start if hasattr(segment, 'start') else 0
        
        if i < total_segments - 1:
            finish_time = segments[i + 1].start if hasattr(segments[i + 1], 'start') else start_time + 1
        else:
            finish_time = segment.end if hasattr(segment, 'end') else start_time + 1
        
        task = asyncio.create_task(process_chunk(text))
        tasks.append((text, start_time, finish_time, task))
    
    # Wait for all tasks to complete
    for text, start_time, finish_time, task in tasks:
        keyword = await task
        lines_with_keywords.append({
            "text": text,
            "keyword": keyword,
            "start": start_time,
            "finish": finish_time
        })
    
    return {
        "transcription": lines_with_keywords
    }

# Modified FastAPI endpoint to use async
@app.post("/process-audio/")
async def process_audio(request: AudioRequest):
    audio_url = request.audio_url
    audio_filename = download_audio(audio_url)
    if audio_filename:
        transcription = transcribe_audio(audio_filename)
        if transcription:
            json_response = await create_json_response(transcription)
            logging.info("Successfully processed audio.")
            os.remove(audio_filename)
            return json_response
        else:
            return {"error": "Transcription failed"}
    else:
        return {"error": "Audio download failed"}

# Start the queue processor
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_queue())

# Main script
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
