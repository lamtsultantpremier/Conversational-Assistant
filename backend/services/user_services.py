from datetime import datetime, timedelta

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

import configs
from services import security
from src.database.database import get_db
from src.models import UsersModel
from src.schemas import CreateUser, CurrentUser, ReadUser, SimpleUser


def create_user(user: CreateUser, db: Session):
    user_db = db.query(UsersModel).filter(UsersModel.email == user.email).first()

    if user_db:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User with this email already exist",
        )

    user.password = security.crypt_password(user.password)

    user_db = UsersModel(**user.model_dump())
    db.add(user_db)
    db.commit()
    db.refresh(user_db)

    return SimpleUser.model_validate(user_db)


def authenticate_user(form_data: OAuth2PasswordRequestForm, db: Session):
    user = db.query(UsersModel).filter(UsersModel.email == form_data.username).first()

    if not user or not security.verify_password(form_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return ReadUser.model_validate(user)


def create_user_token(user: ReadUser):
    payload: dict = {"sub": user.email, "id": user.user_id}
    expires = datetime.utcnow() + timedelta(minutes=30)
    payload.update({"exp": expires})

    return jwt.encode(payload, key=configs.JWT_SECRET, algorithm=configs.JWT_ALGORITHM)


def get_current_user(
    token: str = Depends(security.oauth2_scheme), db: Session = Depends(get_db)
):
    try:

        payload: dict = jwt.decode(
            token, key=configs.JWT_SECRET, algorithms=[configs.JWT_ALGORITHM]
        )
        user_id = payload.get("id")
        user = get_user_by_id(user_id, db)

        return ReadUser.model_validate(user)

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")

    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")


def get_user_by_id(user_id: int, db: Session):
    user = db.query(UsersModel).filter(UsersModel.user_id == user_id).first()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User Not Found"
        )
    return user
