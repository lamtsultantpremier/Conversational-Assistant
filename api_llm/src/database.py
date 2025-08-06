from typing import Annotated

from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import (DeclarativeBase, MappedAsDataclass, Session,
                            sessionmaker)


class Base(DeclarativeBase, MappedAsDataclass):
    pass


engine = create_engine("sqlite:///link_db.db", echo=True)

SessionLocal = sessionmaker(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    except:
        db.close()


DbSession = Annotated[Session, Depends(get_db)]
