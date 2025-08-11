from fastapi import FastAPI, APIRouter, HTTPException, Depends  # type: ignore
from fastapi.responses import StreamingResponse  # type: ignore
import json
from pydantic import BaseModel, ValidationError

from agent.pipeline.stage01_context import generate_context
from agent.pipeline.stage02_extract import extract_knowledge
from agent.pipeline.stage04_slots import propose_slots
from agent.security.jwt_auth import verify_jwt
from agent.monitoring import attach_instrumentator
from agent.models.context import ContextPayload
from agent.slots import Slot
from agent.models.kg import KGPayload
from agent.models.fact import FactIn, Fact, ApproveRequest
from agent.kg.client import Neo4jClient

app = FastAPI(title="Implicit Knowledge Extraction Agent")

router = APIRouter(prefix="/extract", tags=["extract"])
kg_router = APIRouter(prefix="/kg", tags=["kg"])


class ExtractRequest(BaseModel):
    text: str
    focus: str | None = None
    temperature: float = 0.0


class ExtractResponse(KGPayload):
    """Simply returns the extracted knowledge payload."""


@router.post("", response_model=ExtractResponse, dependencies=[Depends(verify_jwt)])
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


@app.post("/stream_pipeline", tags=["stream"], dependencies=[Depends(verify_jwt)])
async def stream_pipeline(req: ExtractRequest) -> StreamingResponse:  # type: ignore[override]
    """Run context → extraction → slot discovery and stream progress via SSE."""

    async def event_generator():  # noqa: WPS430
        # Stage 1: context
        context: ContextPayload = await generate_context(req.text)
        yield {
            "stage": "context",
            "progress": 20,
            "context": context.model_dump(),
        }

        # Stage 2: extraction
        kg_payload = await extract_knowledge(req.text, focus=req.focus, temperature=req.temperature)
        yield {
            "stage": "extract",
            "progress": 60,
            "kg": kg_payload.model_dump(),
        }

        # Stage 4: slot discovery
        slots: list[Slot] = await propose_slots(kg_payload)
        yield {
            "stage": "slots",
            "progress": 80,
            "slots": [s.model_dump() for s in slots],
        }

        yield {"stage": "complete", "progress": 100}

    async def sse_wrapper():
        async for chunk in event_generator():
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    return StreamingResponse(sse_wrapper(), media_type="text/event-stream")


app.include_router(router)


@kg_router.post("/submit", response_model=Fact, dependencies=[Depends(verify_jwt)])
async def kg_submit(fact: FactIn) -> Fact:
    client = Neo4jClient()
    created = client.submit_fact(fact)
    return created


@kg_router.post("/approve", response_model=Fact, dependencies=[Depends(verify_jwt)])
async def kg_approve(payload: ApproveRequest, token: dict = Depends(verify_jwt)) -> Fact:  # type: ignore[override]
    approver = str(token.get("sub", "")) if isinstance(token, dict) else None
    client = Neo4jClient()
    updated = client.approve_fact(payload.fact_id, payload.decision, approver)
    return updated

app.include_router(kg_router)

# Prometheus metrics
try:
    attach_instrumentator(app)
except Exception:  # noqa: BLE001
    pass 