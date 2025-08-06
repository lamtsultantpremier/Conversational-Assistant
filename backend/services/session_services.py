from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.database.database import get_db
from src.models import SessionsModel
from src.schemas import CurrentUser, LLMMessage, ReadSession


def create_session(user: CurrentUser, db: Session):
    session = SessionsModel(user_id=user.user_id)

    db.add(session)
    db.commit()
    db.refresh(session)

    return ReadSession.model_validate(session)


def get_messages_by_session_id(id: int, db: Session = Depends(get_db)):
    session = db.query(SessionsModel).filter(SessionsModel.session_id == id).first()
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not Found"
        )

    return session


def get_session_by_id(session_id: int, db: Session):
    session = (
        db.query(SessionsModel).filter(SessionsModel.session_id == session_id).first()
    )
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not Found"
        )

    return ReadSession.model_validate(session)
