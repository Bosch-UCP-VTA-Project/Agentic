from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from bosch_vta_agentic.utils.schema import AutoTechnicianRAG

app = FastAPI()

# Global variable to store the pipeline
pipeline = None


class QueryRequest(BaseModel):
    query: str


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


@app.on_event("startup")
async def startup_event():
    initialize_pipeline()


@app.post("/query", response_model=ChatResponse)
async def query(request: QueryRequest):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    try:
        # Use request.query to get the query string from the payload
        result = pipeline.query(request.query)

        return ChatResponse(
            answer=result.answer,
            sources=[Source(content=source) for source in result.source_nodes],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
