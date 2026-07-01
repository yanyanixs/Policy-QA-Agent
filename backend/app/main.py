from fastapi import FastAPI
from api.chat_routes import chat_router
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
# Cross-domain is allowed. For production environments,
# please change * to a specific domain name
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
