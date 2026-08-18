from fastapi import FastAPI
from src.routes.triage import router as triage_router

app = FastAPI(title="FlyRank Triage API")

app.include_router(triage_router)
