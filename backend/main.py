from fastapi import FastAPI
from routers import predict

app = FastAPI(
    tỉtle = "Sentiment analysis demo",
)

@app.get("/")
def hello():
    return {"message": "Hello FastAPI"}

app.include_router(predict.router)

