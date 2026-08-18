from fastapi import APIRouter, status, HTTPException
from fastapi.responses import JSONResponse

from src.llm.schema import TriageInput, TriageOutput
from src.llm.service import triage_support_message

router = APIRouter()


@router.post("/triage", response_model=TriageOutput, status_code=status.HTTP_200_OK, summary="Triage a support message")
async def triage_route(payload: TriageInput):
    """Accept a support message and return triage classification (stub during Stage 1 or real LLM in Stage 2)."""
    try:
        result = triage_support_message(payload.text)
    except RuntimeError as e:
        # Do not expose internal details or secrets
        raise HTTPException(status_code=502, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

    # FastAPI will validate/serialize using response_model
    return JSONResponse(status_code=200, content=result.model_dump())
