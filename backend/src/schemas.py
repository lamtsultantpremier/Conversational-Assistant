from typing import List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class SimpleUser(BaseModel):
    nom: str
    prenom: str
    email: EmailStr

    model_config = ConfigDict(from_attributes=True)


class CurrentUser(BaseModel):

    user_id: int
    nom: str
    prenom: str
    email: EmailStr
    sessions: Optional[List["ReadSession"]] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class CreateUser(BaseModel):
    nom: str
    prenom: str
    email: EmailStr
    password: str


class ReadUser(CreateUser):
    user_id: int
    sessions: Optional[List["ReadSession"]] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class CreateSession(BaseModel):
    user_id: int


class ReadSession(CreateSession):
    session_id: int
    messages: Optional[List["ReadMessage"]] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class CreateMessage(BaseModel):
    session_id: int
    role: str
    content: str


class ReadMessage(CreateMessage):
    message_id: int

    model_config = ConfigDict(from_attributes=True)


class LLMMessage(BaseModel):
    role: str
    content: str

    model_config = ConfigDict(from_attributes=True)
