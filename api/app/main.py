from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import predict, load_select_values, training

app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(load_select_values.router)
app.include_router(training.router)