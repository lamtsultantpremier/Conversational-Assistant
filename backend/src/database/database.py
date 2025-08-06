from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

import configs
from src.models import Base

engine = create_engine(url=configs.DATABASE_URL)

Base.metadata.create_all(bind=engine)

LocalSession = scoped_session(sessionmaker(bind=engine))


def get_db():
    db = LocalSession()
    try:
        yield db
    finally:
        db.close()
