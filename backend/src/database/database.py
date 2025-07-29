from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker,scoped_session

from src.models import Base

import configs

engine = create_engine(url = configs.DATABASE_URL)

Base.metadata.create_all(bind = engine)

LocalSession = scoped_session(sessionmaker(bind = engine))

def get_db():
    db = LocalSession()
    try:
        yield db
    finally:
        db.close()