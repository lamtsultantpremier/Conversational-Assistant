from typing import List

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import (DeclarativeBase, Mapped, MappedAsDataclass,
                            mapped_column, relationship)


class Base(DeclarativeBase, MappedAsDataclass):
    pass


class UsersModel(Base):
    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(
        init=False, primary_key=True, autoincrement=True
    )
    nom: Mapped[str] = mapped_column(String(255))
    prenom: Mapped[str] = mapped_column(String(255))
    email: Mapped[str] = mapped_column(String(255))
    password: Mapped[str] = mapped_column(String(255))

    sessions: Mapped[List["SessionsModel"]] = relationship(
        back_populates="user", init=False, default_factory=list
    )


class SessionsModel(Base):
    __tablename__ = "sessions"

    session_id: Mapped[int] = mapped_column(
        init=False, primary_key=True, autoincrement=True
    )
    user_id: Mapped[int] = mapped_column(ForeignKey(column="users.user_id"))

    user: Mapped[UsersModel] = relationship(back_populates="sessions", init=False)
    messages: Mapped[List["MessagesModel"]] = relationship(
        back_populates="session", init=False, default_factory=list
    )


class MessagesModel(Base):
    __tablename__ = "messages"

    message_id: Mapped[int] = mapped_column(
        init=False, primary_key=True, autoincrement=True
    )
    session_id: Mapped[int] = mapped_column(ForeignKey(column="sessions.session_id"))
    role: Mapped[str] = mapped_column(String(255))
    content: Mapped[str] = mapped_column()

    session: Mapped[SessionsModel] = relationship(back_populates="messages", init=False)
