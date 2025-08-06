from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from services import session_services, user_services
from src.database.database import get_db
from src.schemas import CreateSession, LLMMessage, ReadMessage, ReadSession

router = APIRouter()


@router.post("")
def create_session(
    db: Session = Depends(get_db), user=Depends(user_services.get_current_user)
):

    session = session_services.create_session(user, db)
    return session.session_id


@router.get("/{id}/messages")
def get_messages_from_session(
    session: Annotated[
        ReadSession, Depends(session_services.get_messages_by_session_id)
    ],
    user=Depends(user_services.get_current_user),
):
    messages_history = [
        LLMMessage.model_validate(message) for message in session.messages
    ]

    return messages_history
