from fastapi import FastAPI, APIRouter, HTTPException  # type: ignore
from pydantic import BaseModel, ValidationError

from agent.pipeline.stage02_extract import extract_knowledge
from agent.models.kg import KGPayload

app = FastAPI(title="Implicit Knowledge Extraction Agent")

router = APIRouter(prefix="/extract", tags=["extract"])


class ExtractRequest(BaseModel):
    text: str
    focus: str | None = None
    temperature: float = 0.0


class ExtractResponse(KGPayload):
    """Simply returns the extracted knowledge payload."""


@router.post("", response_model=ExtractResponse)
async def extract(req: ExtractRequest) -> ExtractResponse:
    """Endpoint to extract knowledge graph information.

    Handles internal errors gracefully and always returns a well-structured
    ExtractResponse or a relevant HTTP error response.
    """

    try:
        payload = await extract_knowledge(
            text=req.text,
            focus=req.focus,
            temperature=req.temperature,
        )
    except ValidationError as ve:
        # Input or response validation failed – return 422 Unprocessable Entity
        raise HTTPException(status_code=422, detail=ve.errors())
    except Exception as e:  # noqa: BLE001
        # Catch-all for unexpected errors – return 500 Internal Server Error
        raise HTTPException(status_code=500, detail=str(e))

    # Convert KGPayload -> ExtractResponse without intermediate dict dumping
    return ExtractResponse.model_validate(payload)


app.include_router(router) 