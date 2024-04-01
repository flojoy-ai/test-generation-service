from fastapi import FastAPI
from routes.test_generate import test_generate_router

app = FastAPI()

app.include_router(test_generate_router)
