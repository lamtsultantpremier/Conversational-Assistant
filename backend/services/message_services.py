from sqlalchemy.orm import Session
from src.database.database import get_db
from src.schemas import CreateMessage , ReadMessage
from src.models import MessagesModel

def create_message(create_message : CreateMessage , db : Session):
    message = MessagesModel(**create_message.model_dump())
    db.add(message)
    db.commit()
    db.refresh(message)
    
    return ReadMessage.model_validate(message)