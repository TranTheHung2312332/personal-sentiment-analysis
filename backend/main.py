from fastapi import FastAPI
from routers import predict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    tỉtle = "Sentiment analysis demo",
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def hello():
    return {"message": "Hello FastAPI"}

app.include_router(predict.router)

