from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import os
from groq import Groq
from bosch_vta_agentic.utils.schema import AutoTechnicianRAG

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Global variables
pipeline = None
groq_client = None


class Source(BaseModel):
    content: str


class QueryRequest(BaseModel):
    query: str
    session_id: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    history: List[Dict[str, str]]


class AudioResponse(BaseModel):
    answer: str
    transcribed: str


class ChatMessage(BaseModel):
    role: str
    content: str


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


@app.get("/health_check", response_model=ChatResponse)
async def query():
    return ChatResponse(
        answer="Working!",
    )


@app.post("/query", response_model=ChatResponse)
async def query(request: QueryRequest):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    try:
        session_id = request.session_id
        result = pipeline.query(request.query, session_id)
        history = pipeline.get_history(session_id)
        return ChatResponse(
            answer=result.answer,
            session_id=session_id,
            history=history,
        )
        # result = pipeline.query(request.query)
        # return ChatResponse(
        #     answer=result.answer,
        # )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/audio", response_model=AudioResponse)
async def audio_query(audio: UploadFile = File(...)):
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq client not initialized")

    try:
        audio_content = await audio.read()

        translation = groq_client.audio.translations.create(
            file=("recoding.wav", audio_content),
            model="whisper-large-v3",
            prompt="Specify context or spelling",
            response_format="json",
            temperature=0.0,
        )
        print(translation.text)
        query_text = translation.text

        result = pipeline.query(query_text)

        return AudioResponse(
            answer=result.answer,
            transcribed=query_text,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing audio query: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
