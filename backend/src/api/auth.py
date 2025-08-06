from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from services import user_services
from src.database.database import get_db
from src.schemas import CreateUser, CurrentUser, ReadUser

router = APIRouter()


@router.post("/register")
def create_user(user: CreateUser, db: Session = Depends(get_db)):
    user = user_services.create_user(user, db)
    return user


@router.post("/login")
def log_user(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Session = Depends(get_db),
):
    user = user_services.authenticate_user(form_data, db)
    token = user_services.create_user_token(user)

    return {"access_token": token, "token_type": "bearer"}


@router.get("/me")
def read_user_me(current_user=Depends(user_services.get_current_user)):
    return current_user
