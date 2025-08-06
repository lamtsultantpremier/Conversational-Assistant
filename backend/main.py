from fastapi import FastAPI

from src.api import auth, messages, sessions

app = FastAPI()

app.include_router(router=auth.router, prefix="/auth", tags=["user management"])
app.include_router(
    router=sessions.router, prefix="/sessions", tags=["session management"]
)
app.include_router(
    router=messages.router, prefix="/messages", tags=["message management"]
)
