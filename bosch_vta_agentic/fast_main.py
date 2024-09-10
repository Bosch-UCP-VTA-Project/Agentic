from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
import os
import io
import soundfile as sf
from groq import Groq
from bosch_vta_agentic.utils.schema import AutoTechnicianRAG

app = FastAPI()

# Global variables
pipeline = None
groq_client = None


class Source(BaseModel):
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]


def initialize_pipeline():
    global pipeline
    if pipeline is None:
        manuals_path = "./data/technical_manuals"
        online_resources_path = "./data/online_resources"
        index_path = "./indexes"

        pipeline = AutoTechnicianRAG(manuals_path, online_resources_path, index_path)

        if os.path.exists(index_path):
            print("Loaded existing index.")
        else:
            pipeline.load_or_create_indexes()
            pipeline.save_indexes()
            print("Created and saved new index.")


def initialize_groq():
    global groq_client
    if groq_client is None:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        groq_client = Groq(api_key=groq_api_key)


@app.on_event("startup")
async def startup_event():
    initialize_pipeline()
    initialize_groq()


@app.post("/query", response_model=ChatResponse)
async def query(query: str):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    try:
        result = pipeline.query(query)
        return ChatResponse(
            answer=result.answer,
            sources=[Source(content=source) for source in result.source_nodes],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/audio", response_model=ChatResponse)
async def audio_query(audio: UploadFile = File(...)):
    if not pipeline or not groq_client:
        raise HTTPException(
            status_code=500, detail="Pipeline or Groq client not initialized"
        )

    try:
        # Read and process the audio file
        audio_content = await audio.read()
        audio_io = io.BytesIO(audio_content)
        audio_data, sample_rate = sf.read(audio_io)

        # Transcribe using Groq's Whisper model
        transcription_response = groq_client.chat.completions.create(
            model="whisper-large-v3",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that transcribes audio.",
                },
                {
                    "role": "user",
                    "content": f"Transcribe the following audio: {audio_data}",
                },
            ],
        )
        transcribed_text = transcription_response.choices[0].message.content

        # Query the pipeline with the transcribed text
        result = pipeline.query(transcribed_text)

        return ChatResponse(
            answer=result.answer,
            sources=[Source(content=source) for source in result.source_nodes],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing audio query: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
