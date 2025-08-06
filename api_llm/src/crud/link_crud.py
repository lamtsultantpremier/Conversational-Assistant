from typing import List

from sqlalchemy.orm import Session

from src import schemas
from src.models import Link


def create_link(db: Session, url: schemas.LinkCreate) -> Link:
    db_url = Link(**url.model_dump())
    db.add(db_url)
    db.commit()
    return db_url


def get_links(db: Session) -> List[Link]:
    db_links = db.query(Link).all()
    return db_links
