from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="Gold Price Direction Predictor",
    description=(
        "Machine-learning API that predicts whether the next hourly "
        "Gold candle will move up or down."
    ),
    version="0.1.0",
)

app.include_router(router)