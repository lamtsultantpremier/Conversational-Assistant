from fastapi import APIRouter , Depends
from src.schemas import CreateMessage , LLMMessage,ReadUser
from sqlalchemy.orm import Session
from src.database.database import get_db
from services import message_services , session_services , user_services
import configs
import requests


router = APIRouter()

@router.post("")
def create_message(create_message : CreateMessage , 
                   db : Session = Depends(get_db) , 
                   user = Depends(user_services.get_current_user)):

    user_message = message_services.create_message(create_message , db)

    current_message_session = session_services.get_session_by_id(user_message.session_id , db)

    message_history = [LLMMessage.model_validate(message).model_dump() for message in current_message_session.messages]

    payload = {"input" : user_message.content , "messages" : message_history}
    
    response = requests.post(configs.URL_API_LLM , json = payload).json()
    
    llm_message = message_services.create_message(CreateMessage(session_id = user_message.session_id , 
                                                                role = "assistant" , 
                                                                content = response) , db)
    return response