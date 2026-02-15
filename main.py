from fastapi import FastAPI
from app.api.endpoints import router

app = FastAPI(
    title="Iris Predykcje API",
    description="API do klasyfikacji gatunków kwiatów (irysów) na podstawie ich cech",
    version="0.1.0",
)

app.include_router(router)


@app.get("/")
def read_root():
    return {
        "message": "Witaj w API do klasyfikacji gatunków kwiatów (irysów) na podstawie ich cech"
    }
