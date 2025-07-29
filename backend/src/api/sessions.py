from fastapi import APIRouter,Depends
from src.schemas import CreateSession , LLMMessage
from sqlalchemy.orm import Session
from src.database.database import get_db
from services import session_services , user_services
from src.schemas import ReadMessage , ReadSession
from typing import Annotated
router = APIRouter()

@router.post("")
def create_session(db : Session = Depends(get_db),
                   user = Depends(user_services.get_current_user)):
    
    session = session_services.create_session(user,db)
    return session.session_id

@router.get("/{id}/messages")
def get_messages_from_session( session : Annotated [ReadSession , Depends(session_services.get_messages_by_session_id)],
                              user = Depends(user_services.get_current_user)):
    messages_history = [LLMMessage.model_validate(message) for message in session.messages]
                             
    return messages_history